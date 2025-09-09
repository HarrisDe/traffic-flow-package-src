import math
import argparse
import utils
import model
import time
import datetime
import numpy as np
import pandas as pd
import os

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--P', type=int, default=12,
                    help='history steps')
parser.add_argument('--Q', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=3,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=5,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='../data/PEMS-BAY/PEMS.csv',
                    help='traffic file')
parser.add_argument('--SE_file', default='../data/PEMS-BAY/SE(PeMS).txt',
                    help='spatial emebdding file')
parser.add_argument('--model_file', default='results/GMAN(PeMS)',
                    help='save the model to disk')
parser.add_argument('--log_file', default='results/log(PeMS)',
                    help='log file')
parser.add_argument('--metrics_file', help='metrics file')
args = parser.parse_args()


# Define directory path
results_dir = "results/third_experimental_study"

# Create the directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)


start = time.time()

log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10: -1])

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
 mean, std) = utils.loadData(args)
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')


# Load dataset to extract timestamps & sensor names
# Read dataset to get timestamps
df = pd.read_csv(args.traffic_file, index_col=0)
sensor_ids = df.columns.tolist()  # Get sensor names
print(f"min timestamp: {df.index.min()}, max timestamp: {df.index.max()}")
# Ensure sensor names match number of sensors
if len(sensor_ids) != testX.shape[2]:
    raise ValueError(
        f"Mismatch: Found {len(sensor_ids)} sensors, but expected {testX.shape[2]} from test data.")

try:  # Extract timestamps corresponding to testX
    # Get timestamps corresponding to testX
    timestamps = df.index[len(trainX): len(trainX) + len(testX) + args.Q]
except Exception as e:
    print("An error occurred:", e)

# train model
utils.log_string(log, 'compiling model...')
T = 24 * 60 // args.time_slot
num_train, _, N = trainX.shape
X, TE, label, is_training = model.placeholder(args.P, args.Q, N)
global_step = tf.Variable(0, trainable=False)
bn_momentum = tf.compat.v1.train.exponential_decay(
    0.5, global_step,
    decay_steps=args.decay_epoch * num_train // args.batch_size,
    decay_rate=0.5, staircase=True)
bn_decay = tf.minimum(0.99, 1 - bn_momentum)
pred = model.GMAN(
    X, TE, SE, args.P, args.Q, T, args.L, args.K, args.d,
    bn=True, bn_decay=bn_decay, is_training=is_training)
pred = pred * std + mean
loss = model.mae_loss(pred, label)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
learning_rate = tf.compat.v1.train.exponential_decay(
    args.learning_rate, global_step,
    decay_steps=args.decay_epoch * num_train // args.batch_size,
    decay_rate=0.7, staircase=True)
learning_rate = tf.maximum(learning_rate, 1e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    # parameters += np.product([x.value for x in variable.get_shape()])
    parameters += np.product([x for x in variable.get_shape()])
utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: trainX[start_idx: end_idx],
            TE: trainTE[start_idx: end_idx],
            label: trainY[start_idx: end_idx],
            is_training: True}
        _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
        train_loss += loss_batch * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()
    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: valX[start_idx: end_idx],
            TE: valTE[start_idx: end_idx],
            label: valY[start_idx: end_idx],
            is_training: False}
        loss_batch = sess.run(loss, feed_dict=feed_dict)
        val_loss += loss_batch * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    utils.log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
         args.max_epoch, end_train - start_train, end_val - start_val))
    utils.log_string(
        log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
            log,
            'val loss decrease from %.4f to %.4f, saving model to %s' %
            (val_loss_min, val_loss, args.model_file))
        wait = 0
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
    else:
        wait += 1

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
num_test = testX.shape[0]
print(f"num_test: {num_test}")
trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: trainX[start_idx: end_idx],
        TE: trainTE[start_idx: end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict=feed_dict)
    trainPred.append(pred_batch)
trainPred = np.concatenate(trainPred, axis=0)
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: valX[start_idx: end_idx],
        TE: valTE[start_idx: end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict=feed_dict)
    valPred.append(pred_batch)
valPred = np.concatenate(valPred, axis=0)
testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()

# Initialize a list to store batch results
batch_results = []
# Load dataset to extract timestamps & sensor names
# Read dataset to get timestamps
df = pd.read_csv(args.traffic_file, index_col=0)
print(f"df shape: {df.shape}")
df.index = pd.to_datetime(df.index)  # Ensure index is datetime
sensor_ids = df.columns.tolist()  # Get sensor names
# Ensure sensor names match number of sensors
if len(sensor_ids) != testX.shape[2]:
    raise ValueError(
        f"Mismatch: Found {len(sensor_ids)} sensors, but expected {testX.shape[2]} from test data.")
# Get timestamps corresponding to testX
# timestamps = df.index[len(trainX): len(trainX) + len(testX) + args.Q]
# Extract timestamps correctly **for testY predictions**
# Start timestamps from Pth step + Q
# Extract timestamps that directly correspond to testY
# Directly get the timestamps for testY
# works but not with batches aligned correctly
timestamps_testY = df.index[-len(testY) + args.P + 1:]

print(f"Extracted {len(timestamps_testY)} timestamps for predictions.")
print(f"testX shape: {testX.shape}")
print(f"testY shape: {testY.shape}")
# Ensure the number of timestamps matches testY.shape[0]
# if len(timestamps_testY) != testY.shape[0] + args.P + 1:
#     raise ValueError(f"Timestamp mismatch: {len(timestamps_testY)} timestamps found, but expected {testY.shape[0] + args.P + 1}.")

timestamps_testY = utils.loadData(args, output_timestamps=True)
# orig for loop
for batch_idx in range(num_batch):
    # from the utils also extract the timestamps of the test set. Use other start and end_ind inside the other loops in order to get the correct timestamps, and values
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: testX[start_idx: end_idx],
        TE: testTE[start_idx: end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict=feed_dict)
    # print(f"pred_batch shape: {pred_batch.shape}")
    actual_batch = testY[start_idx: end_idx]
    # print(f"actual_batch shape: {actual_batch.shape}") (32,15,204)
    testPred.append(pred_batch)
    timestamps_batch = timestamps_testY[start_idx: end_idx]
    # print(f'timestamps_batch of index {batch_idx} shape: {timestamps_batch.shape}') (32,15,1)


# works but (batch_size = 32(default)*Q = 32*15 = 480)
    # Store results for each predicted timestep (1 to Q)
    for i in range(len(pred_batch)):  # Iterate over samples in batch
        # Assign unique sample number (1-based index)
        sample_nr = start_idx + i + 1
        for t in range(args.Q):  # Iterate over timesteps
            # Ensure timestamp index is within bounds
            # timestamp_idx = min(start_idx + i + t, len(timestamps_testY) - 1)
            # timestamp_idx = sample_nr + t + args.P
            # timestamp_idx = start_idx + t
            timestamp = timestamps_batch[i, t]
            timestamp = utils.convert_timestamp_data(timestamp)
            # Dictionary to store current row data
            row_data = {'batch_nr': batch_idx + 1,
                        'sample_nr': sample_nr,
                        # 'timestamp': timestamps_testY[timestamp_idx],
                        'timestamp': timestamp,
                        'timestep': t + 1,
                        }  # Batch number & timestamp

            # Add actual and predicted values for each sensor
            for s, sensor_name in enumerate(sensor_ids):
                row_data[sensor_name] = actual_batch[i, t, s]  # Actual value
                # Predicted value
                row_data[f"{sensor_name}_pred"] = pred_batch[i, t, s]

            # Append row to batch results
            batch_results.append(row_data)
# doesn't match timestamps properly

    # Store results for Q=15 predicted timesteps
    # for t in range(args.Q):  # Iterate over timesteps
    #     #timestamp_idx = start_idx + t
    #     timestamp_idx = start_idx + t + args.P + 1
    #     if timestamp_idx >= len(timestamps_testY):
    #         continue  # Skip out-of-bounds timestamps

    #     # Dictionary to store current row data
    #     row_data = {
    #         'batch_nr': batch_idx + 1,  # **Ensuring batch_nr appears Q=15 times per batch**
    #         'timestamp': timestamps_testY[timestamp_idx],  # **Correct timestamp**
    #         'timestep': t + 1  # Prediction step (1 to Q)
    #     }

    #     # Add actual and predicted values for each sensor at each predicted timestep
    #     for s, sensor_name in enumerate(sensor_ids):
    #         row_data[sensor_name] = actual_batch[:, t, s].mean()  # Store mean across batch (corrected)
    #         row_data[f"{sensor_name}_pred"] = pred_batch[:, t, s].mean()  # Store mean across batch

    #     # Append row to batch results
    #     batch_results.append(row_data)

################################################################################
end_test = time.time()

# Convert batch results to a DataFrame
batch_results_df = pd.DataFrame(batch_results)

# Save DataFrame to CSV

batch_results_file = f'{results_dir}/test_results_Q_{args.Q}_from_bash.csv'
print(f"batch_results_df saved in: {batch_results_file}")
batch_results_df.to_csv(batch_results_file, index=False)

utils.log_string(
    log, f'Test results FOR ALL DATA saved to {batch_results_file}')


testPred = np.concatenate(testPred, axis=0)
train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)

utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'performance in each prediction step')
MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
utils.log_string(
    log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
    (average_mae, average_rmse, average_mape * 100))
end = time.time()

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame({'MAE': MAE, 'RMSE': RMSE, 'MAPE': MAPE})
metrics_df.to_csv(args.metrics_file, index=False)

utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))

# Additional Test: Check the Last Timestep Only
utils.log_string(log, '**** testing model (last timestep only) ****')
print('test prediction shape:', testPred.shape)
print('test ground truth shape:', testY.shape)
# Extract predictions for the last timestep
last_timestep_predictions = testPred[:, -1]
# Extract ground truth for the last timestep
last_timestep_ground_truth = testY[:, -1]

# Compute metrics for the last timestep
last_mae, last_rmse, last_mape = utils.metric(
    last_timestep_predictions, last_timestep_ground_truth)


# Compute Median Absolute Error (MedianAE)
last_medae = np.median(
    np.abs(last_timestep_predictions - last_timestep_ground_truth))

# Log metrics for the last timestep
utils.log_string(log, 'Metrics for the last timestep:')
utils.log_string(log, 'Last Timestep - MAE: %.2f, MedianAE: %.2f, RMSE: %.2f, MAPE: %.2f%%' %
                 (last_mae, last_medae, last_rmse, last_mape * 100))

# Save last timestep metrics to CSV
last_timestep_df = pd.DataFrame({
    'Metric': ['MAE', 'MedianAE', 'RMSE', 'MAPE'],
    'Value': [last_mae, last_medae, last_rmse, last_mape]
})
last_timestep_df.to_csv('results/last_timestep_metrics.csv', index=False)
utils.log_string(
    log, 'Last timestep metrics saved to results/last_timestep_metrics.csv')
# Saving sensor-level results for the last timestep
utils.log_string(log, 'Saving sensor-level results for the last timestep...')

# Extract sensor IDs from the traffic file
# Skip the first column if it's timestamps
sensor_ids = pd.read_csv(args.traffic_file, nrows=1).columns[1:]
# Read full dataset with timestamps
df = pd.read_csv(args.traffic_file, index_col=0)

# Ensure timestamps are correctly aligned with the last historical timestep (Pth step)
# Extract timestamps corresponding to the Pth step
timestamps = df.index[len(trainX): len(trainX) + len(testX)]
print(f"Extracted {len(timestamps)} timestamps for predictions.")

# old
# timestamps = pd.read_csv(args.traffic_file, index_col=0).index[-len(testX):]  # Extract the last timestamps matching testX
# Verify that the number of timestamps matches the number of predictions
if len(timestamps) != last_timestep_ground_truth.shape[0]:
    raise ValueError(
        f"Mismatch: {len(timestamps)} timestamps found, but expected {last_timestep_ground_truth.shape[0]}.")

print('sensor ids:', sensor_ids)
if len(sensor_ids) != last_timestep_ground_truth.shape[1]:
    raise ValueError(
        f"Mismatch: {len(sensor_ids)} sensor IDs found, but expected {last_timestep_ground_truth.shape[1]}.")

# Create DataFrame for sensor-level details
# sensor_results_df = pd.DataFrame({
#     'Sensor_ID': sensor_ids,
#     # Use the last timestep across sensors
#     'y_act': last_timestep_ground_truth[-1],
#     'y_pred': last_timestep_predictions[-1]
# })

# Create DataFrame for sensor-level details
sensor_results_df = pd.DataFrame({
    # Repeat each sensor ID across all timestamps
    'sensor_id': np.repeat(sensor_ids, len(timestamps)),
    # Tile timestamps for all sensors
    'timestamp': np.tile(timestamps, len(sensor_ids)),
    'y_act': last_timestep_ground_truth.flatten(),
    'y_pred': last_timestep_predictions.flatten(),
})
# Sort the DataFrame by sensor_id first, then timestamp
sensor_results_df = sensor_results_df.sort_values(
    by=['sensor_id', 'timestamp']).reset_index(drop=True)

# Save to CSV
sensor_results_file = f'results/sensor_last_timestep_Q_{args.Q}.csv'
sensor_results_df.to_csv(sensor_results_file, index=False)
utils.log_string(log, f'Sensor-level results saved to {sensor_results_file}')


sess.close()
log.close()
