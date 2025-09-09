# Absolute path to 'src/GMAN' directory
import datetime
import numpy as np
import pandas as pd
from argparse import Namespace
import gc
import argparse
import math
import glob
import tensorflow as tf
import time
import os
import sys
import random



# gman_path = os.path.abspath("src/GMAN")
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"repo_root: {repo_root}")
gman_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src', 'GMAN'))
sys.path.append(gman_path)
gman_path = os.path.abspath("src/GMAN")
sys.path.append(gman_path)
print(f'working directory: {os.getcwd()}')
print(f"gman_path: {gman_path}")
tf.compat.v1.disable_eager_execution()
# Print the current working directory to verify module search paths
print("Current working directory:", os.getcwd())
import utils
import model

def set_random_seed(seed: int = 42, use_gpu: bool = True):
    """
    Set the random seed for Python, NumPy, and TensorFlow (TF 1.x) 
    to ensure experiment reproducibility.

    Optionally disables GPU usage for full determinism.

    Args:
        seed (int): The random seed to use for reproducibility.
        use_gpu (bool): 
            If False, disables GPU usage by setting CUDA_VISIBLE_DEVICES = '-1'.
            This is useful for achieving completely deterministic behavior.
            If True (default), uses available GPU but may introduce some nondeterminism.

    Usage:
        >>> from utils import set_random_seed
        >>> set_random_seed(seed=123, use_gpu=True)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU disabled for reproducibility (CUDA_VISIBLE_DEVICES = -1)")
    else:
        print("GPU enabled. Note: some TensorFlow ops may still be non-deterministic.")


def load_data(args, log):
    """
    Load and preprocess the dataset.

    Args:
        args (Namespace): Command-line arguments or configuration parameters.
        log (file object): Log file to record progress and messages.

    Returns:
        tuple: (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std)
    """
    if log is not None:
        utils.log_string(log, 'loading data...')
    # Load data using the utils.loadData function
    (trainX, trainTE, trainY,
     valX, valTE, valY,
     testX, testTE, testY,
     SE, mean, std) = utils.loadData(args)
    # Log the shapes of the loaded data
    if log is not None:
        utils.log_string(
            log, f'trainX: {trainX.shape}\ttrainY: {trainY.shape}')
        utils.log_string(log, f'valX: {valX.shape}\tvalY: {valY.shape}')
        utils.log_string(log, f'testX: {testX.shape}\ttestY: {testY.shape}')
        utils.log_string(log, 'data loaded!')
    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std


def build_gman(args, SE, T, N, log, num_train, mean, std):
    """
    Build the GMAN model.

    Args:
        args (Namespace): Command-line arguments or configuration parameters.
        SE (np.ndarray): Spatial embeddings.
        T (int): Number of time steps in a day.
        N (int): Number of sensors or nodes.
        log (file object): Log file to record progress and messages.
        num_train (int): Number of training samples.
        mean (float): Mean value for scaling.
        std (float): Standard deviation for scaling.

    Returns:
        tuple: (X, TE, label, is_training, pred, loss, train_op, global_step)
    """
    if log is not None:
        utils.log_string(log, 'compiling model...')
    # Define placeholders for input data
    X, TE, label, is_training = model.placeholder(args.P, args.Q, N)
    # Define global step for learning rate decay
    global_step = tf.Variable(0, trainable=False)
    # Define batch normalization momentum with exponential decay
    bn_momentum = tf.compat.v1.train.exponential_decay(
        0.5, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.5,
        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    # Build the GMAN model with batch normalization
    pred = model.GMAN(X, TE, SE, args.P, args.Q, T, args.L, args.K, args.d,
                      bn=True, bn_decay=bn_decay, is_training=is_training)
    # Scale predictions using mean and std
    pred = pred * std + mean
    # Define loss function (Mean Absolute Error)
    loss = model.mae_loss(pred, label)
    # Add predictions and loss to TensorFlow collections
    tf.compat.v1.add_to_collection('pred', pred)
    tf.compat.v1.add_to_collection('loss', loss)
    # Define learning rate with exponential decay
    learning_rate = tf.compat.v1.train.exponential_decay(
        args.learning_rate, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.7,
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-5)
    # Define optimizer (Adam) and training operation
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    # Calculate the number of trainable parameters
    parameters = sum(np.product(variable.get_shape().as_list())
                     for variable in tf.compat.v1.trainable_variables())
    if log is not None:
        utils.log_string(log, f'trainable parameters: {parameters:,}')
        utils.log_string(log, 'model compiled!')
    print(f"GMAN model compiled with {parameters:,} trainable parameters.")
    return X, TE, label, is_training, pred, loss, train_op, global_step


def train_gman(args, sess, X, TE, label, is_training, train_op, loss,
               trainX, trainTE, trainY, valX, valTE, valY, log):
    """
    Train the GMAN model.

    Args:
        args (Namespace): Command-line arguments or configuration parameters.
        sess (tf.Session): TensorFlow session.
        X (tf.Tensor): Placeholder for input features.
        TE (tf.Tensor): Placeholder for temporal embeddings.
        label (tf.Tensor): Placeholder for ground truth labels.
        is_training (tf.Tensor): Placeholder for training mode.
        train_op (tf.Operation): Training operation.
        loss (tf.Tensor): Loss function.
        trainX (np.ndarray): Training input data.
        trainTE (np.ndarray): Training temporal embeddings.
        trainY (np.ndarray): Training ground truth labels.
        valX (np.ndarray): Validation input data.
        valTE (np.ndarray): Validation temporal embeddings.
        valY (np.ndarray): Validation ground truth labels.
        log (file object): Log file to record progress and messages.

    Returns:
        tuple: (best_epoch, start_train, end_train)
    """
    utils.log_string(log, '**** training model ****')
    print("Starting training...")
    num_train, _, N = trainX.shape
    num_val = valX.shape[0]
    wait = 0
    val_loss_min = np.inf
    best_epoch = 0
    saver = tf.compat.v1.train.Saver()  # Saver for saving the model

    start_train = time.time()
    for epoch in range(args.max_epoch):
        start_train_epoch = time.time()
        if wait >= args.patience:
            utils.log_string(log, f'early stop at epoch: {epoch:04d}')
            print(f"Early stopping at epoch {epoch}.")
            break

        # Shuffle training data
        permutation = np.random.permutation(num_train)
        trainX, trainTE, trainY = trainX[permutation], trainTE[permutation], trainY[permutation]
        #start_train = time.time()
        train_loss = 0
        num_batch = math.ceil(num_train / args.batch_size)

        # Training loop
        print(f"Epoch {epoch + 1}/{args.max_epoch}: Starting training...")
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            feed_dict = {
                X: trainX[start_idx: end_idx],
                TE: trainTE[start_idx: end_idx],
                label: trainY[start_idx: end_idx],
                is_training: True
            }
            _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_batch * (end_idx - start_idx)
        train_loss /= num_train
        # end_train = time.time()
        end_train_epoch = time.time()

        # Validation loop
        print(f"Epoch {epoch + 1}/{args.max_epoch}: Starting validation...")
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
                is_training: False
            }
            loss_batch = sess.run(loss, feed_dict=feed_dict)
            val_loss += loss_batch * (end_idx - start_idx)
        val_loss /= num_val
        end_val = time.time()

        # Log training and validation metrics
        utils.log_string(log,
                         f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                         f'epoch: {epoch + 1:04d}/{args.max_epoch}, '
                         f'training time per epoch: {end_train_epoch - start_train_epoch:.1f}s, '
                         f'inference time: {end_val - start_val:.1f}s')
        utils.log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        print(
            f"Epoch {epoch + 1}/{args.max_epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss <= val_loss_min:
            utils.log_string(log,
                             f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, '
                             f'saving model to {args.model_file}')
            print(
                f"Validation loss improved from {val_loss_min:.4f} to {val_loss:.4f}. Saving model...")
            wait = 0
            val_loss_min = val_loss
            saver.save(sess, args.model_file)
            best_epoch = epoch + 1
        else:
            wait += 1
            print(
                f"Validation loss did not improve. Patience: {wait}/{args.patience}")
            
            
    end_train = time.time()

    print(f"Training completed. Best epoch: {best_epoch}")
    return best_epoch, start_train, end_train, start_val, end_val


def test_gman(args, sess, X, TE, is_training, pred, testX, testTE, testY, log):
    """
    Test the GMAN model.

    Args:
        args (Namespace): Command-line arguments or configuration parameters.
        sess (tf.Session): TensorFlow session.
        X (tf.Tensor): Placeholder for input features.
        TE (tf.Tensor): Placeholder for temporal embeddings.
        is_training (tf.Tensor): Placeholder for training mode.
        pred (tf.Tensor): Model predictions.
        testX (np.ndarray): Test input data.
        testTE (np.ndarray): Test temporal embeddings.
        testY (np.ndarray): Test ground truth labels.
        log (file object): Log file to record progress and messages.

    Returns:
        tuple: (testPred, start_test, end_test, batch_results_df)
    """
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
    saver.restore(sess, args.model_file)
    num_test = testX.shape[0]
    testPred = []
    batch_results = []
    num_batch = math.ceil(num_test / args.batch_size)
    # Read dataset timestamps to get the corresponding timestamps for testY
    timestamps_testY = utils.loadData(args, output_timestamps=True)
    df = pd.read_csv(args.traffic_file, index_col=0)
    sensor_ids = df.columns.tolist()
    start_test = time.time()
    # Testing loop
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: testX[start_idx: end_idx],
            TE: testTE[start_idx: end_idx],
            is_training: False
        }
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        actual_batch = testY[start_idx: end_idx]
        testPred.append(pred_batch)
        timestamps_batch = timestamps_testY[start_idx: end_idx]
        for i in range(len(pred_batch)):
            sample_nr = start_idx + i + 1
            for t in range(args.Q):
                timestamp = timestamps_batch[i, t]
                timestamp = utils.convert_timestamp_data(timestamp)
                # Dictionary to store current row data
                row_data = {
                    'batch_nr': batch_idx + 1,
                    'sample_nr': sample_nr,
                    'timestamp': timestamp,
                    'timestep': t + 1,
                }
                # Add actual and predicted values for each sensor
                for s, sensor_name in enumerate(sensor_ids):
                    row_data[sensor_name] = actual_batch[i, t, s]
                    row_data[f"{sensor_name}_pred"] = pred_batch[i, t, s]
                batch_results.append(row_data)
    testPred = np.concatenate(testPred, axis=0)
    end_test = time.time()
    batch_results_df = pd.DataFrame(batch_results)
    return testPred, start_test, end_test, batch_results_df


def test_gman_opt(args, sess, X, TE, is_training, pred, testX, testTE, testY, log):
    """
    Optimized version of the GMAN testing function with batch-wise data handling.

    Args:
        args (Namespace): Command-line arguments or configuration parameters.
        sess (tf.Session): TensorFlow session.
        X (tf.Tensor): Placeholder for input features.
        TE (tf.Tensor): Placeholder for temporal embeddings.
        is_training (tf.Tensor): Placeholder for training mode.
        pred (tf.Tensor): Model predictions.
        testX (np.ndarray): Test input data.
        testTE (np.ndarray): Test temporal embeddings.
        testY (np.ndarray): Test ground truth labels.
        log (file object): Log file to record progress and messages.

    Returns:
        tuple: (pred_flat, start_test, end_test, batch_results_df)
    """
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, f'loading model from {args.model_file}')
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
    saver.restore(sess, args.model_file)
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')

    num_test = testX.shape[0]
    num_sensors = testY.shape[2]
    num_timesteps = testY.shape[1]

    if args.traffic_file is not None:
        sensor_ids = pd.read_csv(
            args.traffic_file, index_col=0).columns.tolist()
    else:
        sensor_ids = args.df_gman.drop('test_set', axis=1).columns.tolist()

    # Load timestamps and reshape to match batch structure
    timestamps_testY = utils.loadData_test_set_as_input_column(
        args, output_timestamps=True).reshape(num_test, num_timesteps)

    start_test = time.time()

    # Lists to store predictions, actual values, timestamps, and batch numbers
    all_pred = []
    all_actual = []
    all_timestamps = []
    all_batch_nrs = []

    num_batch = math.ceil(num_test / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            is_training: False
        }
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        actual_batch = testY[start_idx:end_idx]
        timestamps_batch = timestamps_testY[start_idx:end_idx]
        batch_numbers = np.full(
            (actual_batch.shape[0] * num_timesteps,), batch_idx + 1)
        all_pred.append(pred_batch.reshape(-1, num_sensors))
        all_actual.append(actual_batch.reshape(-1, num_sensors))
        all_timestamps.append(timestamps_batch.flatten())
        all_batch_nrs.append(batch_numbers)
    end_test = time.time()

    # Concatenate results from all batches
    pred_flat = np.concatenate(all_pred, axis=0)
    actual_flat = np.concatenate(all_actual, axis=0)
    timestamps_flat = np.concatenate(all_timestamps)
    batch_nrs_flat = np.concatenate(all_batch_nrs)

    # Convert timestamps using vectorized function
    timestamps_converted = np.vectorize(
        utils.convert_timestamp_data)(timestamps_flat)

    # Construct DataFrame with the results
    data_dict = {
        'batch_nr': batch_nrs_flat,
        'sample_nr': np.repeat(np.arange(1, num_test + 1), num_timesteps),
        'timestep': np.tile(np.arange(1, num_timesteps + 1), num_test),
        'timestamp': timestamps_converted
    }
    for s, sensor_name in enumerate(sensor_ids):
        data_dict[sensor_name] = actual_flat[:, s]
        data_dict[f"{sensor_name}_pred"] = pred_flat[:, s]

    batch_results_df = pd.DataFrame(data_dict)

    return pred_flat, start_test, end_test, batch_results_df


def log_metrics(log, testPred, testY, args, start_test, end_test,
                trainPred=None, trainY=None, valPred=None, valY=None):
    """
    Log evaluation metrics for the predictions.

    Args:
        log (file object): Log file for messages.
        testPred (np.ndarray): Test predictions.
        testY (np.ndarray): Test ground truth labels.
        args (Namespace): Configuration parameters.
        start_test (float): Start time of testing.
        end_test (float): End time of testing.
        trainPred (np.ndarray, optional): Training predictions.
        trainY (np.ndarray, optional): Training ground truth.
        valPred (np.ndarray, optional): Validation predictions.
        valY (np.ndarray, optional): Validation ground truth.

    Returns:
        tuple: (last_AE, last_RMSE, last_MAPE)
    """
    print(f"Test predictions shape in log_metrics: {testPred.shape}")
    print(f"Test labels shape in log_metrics: {testY.shape}")
    if trainPred is not None:
        train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
    if valPred is not None:
        val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
    test_mae, test_rmse, test_mape = utils.metric(testPred, testY)

    # Log testing time and overall metrics
    utils.log_string(log, f'testing time: {end_test - start_test:.1f}s')
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    if trainPred is not None:
        utils.log_string(
            log, f'train            {train_mae:.2f}\t\t{train_rmse:.2f}\t\t{train_mape * 100:.2f}%')
    if valPred is not None:
        utils.log_string(
            log, f'val              {val_mae:.2f}\t\t{val_rmse:.2f}\t\t{val_mape * 100:.2f}%')
    utils.log_string(
        log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape * 100:.2f}%')

    # Calculate and log metrics for each prediction step
    MAE, RMSE, MAPE = [], [], []
    # print('*************Calculating with utils.metric_unweighted*************')
    # for q in range(args.Q):
    #     mae, rmse, mape = utils.metric_unweighted(testPred[:, q], testY[:, q])
    #     MAE.append(mae)
    #     RMSE.append(rmse)
    #     MAPE.append(mape)
    #     utils.log_string(
    #         log, f'step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%')
    print('*************Calculating with utils.metric*************')
    for q in range(args.Q):
        mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        utils.log_string(
            log, f'step: {q + 1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape * 100:.2f}%')

    # Average metrics over all steps
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    utils.log_string(log,
                     f'average:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape * 100:.2f}%')

    # Last timestep metrics
    last_pred = testPred[:, -1, :]
    last_labels = testY[:, -1, :]
    last_AE = np.abs(last_pred - last_labels)
    last_RMSE = np.sqrt(np.mean(np.square(last_AE)))
    last_smape_values = 2.0 * last_AE / (np.abs(last_pred) + np.abs(last_labels) + 1e-6)
    last_smape = np.mean(last_smape_values)
    last_MAPE_values = np.abs(last_AE / last_labels)
    #last_MAPE = np.mean(last_AE / last_labels)
    last_MAPE = np.mean(last_MAPE_values)

    avg_last_MAE = np.mean(last_AE)
    avg_last_RMSE = last_RMSE
    avg_last_MAPE = np.mean(last_MAPE)

    utils.log_string(log, 'Last Timestep Metrics:')
    utils.log_string(
        log, f'last timestep    {avg_last_MAE:.2f}\t\t{avg_last_RMSE:.2f}\t\t{avg_last_MAPE * 100:.2f}%')

    return last_AE, last_RMSE, last_MAPE, last_MAPE_values, last_smape, last_smape_values


def create_results_df(args, testPred, testY):
    """
    Create a DataFrame containing predictions and ground truth for the last timestep.

    Args:
        testPred (np.ndarray): Test predictions.
        testY (np.ndarray): Test ground truth labels.
    """
    # Extract predictions and ground truth for the last timestep
    last_timestep_predictions = testPred[:, -1]
    last_timestep_ground_truth = testY[:, -1]
    # Read sensor IDs from the traffic file (skipping timestamp column)
    sensor_ids = pd.read_csv(args.traffic_file, nrows=1).columns[1:]
    # Read the full dataset to get timestamps
    df = pd.read_csv(args.traffic_file, index_col=0)
    # Create DataFrame for sensor-level details
    sensor_results_df = pd.DataFrame({
        'sensor_id': np.repeat(sensor_ids, len(timestamps)),
        'timestamp': np.tile(timestamps, len(sensor_ids)),
        'y_act': last_timestep_ground_truth.flatten(),
        'y_pred': last_timestep_predictions.flatten(),
    })
    # Sort by sensor_id and timestamp
    sensor_results_df = sensor_results_df.sort_values(
        by=['sensor_id', 'timestamp']).reset_index(drop=True)


def measure_gman_test_time(args, sess, X, TE, is_training, pred, testX, testTE, log):
    """
    Measure the inference time of the GMAN model without storing results.

    Args:
        args (Namespace): Configuration parameters.
        sess (tf.Session): TensorFlow session.
        X (tf.Tensor): Placeholder for input features.
        TE (tf.Tensor): Placeholder for temporal embeddings.
        is_training (tf.Tensor): Training mode indicator.
        pred (tf.Tensor): Model predictions.
        testX (np.ndarray): Test input data.
        testTE (np.ndarray): Test temporal embeddings.
        log (file object): Log file.

    Returns:
        tuple: (testPred, test_duration)
    """
    utils.log_string(log, '**** measuring GMAN test time ****')
    utils.log_string(log, f'loading model from {args.model_file}')
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
    saver.restore(sess, args.model_file)
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'starting inference...')

    num_test = testX.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)
    testPred = []
    start_test = time.time()

    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            is_training: False
        }
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        testPred.append(pred_batch)

    testPred = np.concatenate(testPred, axis=0)
    end_test = time.time()
    test_duration = end_test - start_test
    utils.log_string(
        log, f'Inference completed in {test_duration:.4f} seconds.')
    return testPred, test_duration


# def save_results(args, results_dir, results_filename, AE, RMSE, MAPE,
#                  best_epoch, start_train, end_train, start_test, end_test, file_name,last_mape_values):
#     """
#     Save evaluation results to a CSV file in the specified results directory.

#     Args:
#         args (Namespace): Configuration parameters.
#         results_dir (str): Directory for saving results.
#         results_filename (str): Filename for the CSV results.
#         AE (list): Mean Absolute Error for each prediction step.
#         RMSE (list): Root Mean Squared Error for each prediction step.
#         MAPE (list): Mean Absolute Percentage Error for each prediction step.
#         best_epoch (int): Best training epoch.
#         start_train (float): Training start time.
#         end_train (float): Training end time.
#         start_test (float): Testing start time.
#         end_test (float): Testing end time.
#     """
#     results_file = os.path.join(results_dir, results_filename)
#     # Create file if it doesn't exist
#     if not os.path.exists(results_file):
#         df = pd.DataFrame(columns=[
#             'P', 'Q', 'MAE', 'MedianAE', 'RMSE', 'MAPE',
#             'MAE_std', 'MedianAE_std', 'RMSE_std', 'MAPE_std',
#             'learning_rate', 'training_time', 'inference_time',
#             'L', 'K', 'best_epoch', 'test_start', 'test_end'
#         ])
#         df.to_csv(results_file, index=False)
#     # Create a new row with the metrics
#     new_row = {
#         'filename': args.filename,
#         'train_ratio': args.train_ratio,
#         'smooth_speeds': args.smooth_speeds,
#         'filter_on_train_only': args.filter_on_train_only,
#         'window_size': args.window_size,
#         'P': args.P,
#         'Q': args.Q,
#         'MAE': np.mean(AE),
#         'MedianAE': np.median(AE),
#         'RMSE': np.sqrt(np.mean(AE**2)),
#         'MAPE': np.mean(MAPE) * 100,
#         'MAE_std': np.std(AE),
#         'MedianAE_std': np.std(AE),
#         'MAPE_std': np.std(last_mape_values) * 100,
#         'learning_rate': args.learning_rate,
#         'training_time': end_train - start_train,
#         'inference_time': end_test - start_test,
#         'L': args.L,
#         'K': args.K,
#         'best_epoch': best_epoch,
#         'test_start': datetime.datetime.fromtimestamp(start_test).strftime('%Y-%m-%d %H:%M:%S'),
#         'test_end': datetime.datetime.fromtimestamp(end_test).strftime('%Y-%m-%d %H:%M:%S')
#     }
#     # Append the new row to the CSV file
#     df = pd.read_csv(results_file)
#     df = df.append(new_row, ignore_index=True)
#     df.to_csv(results_file, index=False)
#     print(f"Metrics saved to {results_file}")


def save_results(args, results_dir, results_filename, AE, RMSE, MAPE, SMAPE,
                 best_epoch, start_train, end_train,start_val, end_val, start_test, end_test, file_name, last_mape_values,last_smape_values):
    """
    Save evaluation results to a CSV file in a consistent column order.

    Args:
        args (Namespace): Configuration parameters.
        results_dir (str): Directory for saving results.
        results_filename (str): Filename for the CSV results.
        AE (list): Absolute Errors for each sample and step.
        RMSE (list): RMSE values.
        MAPE (list): MAPE values.
        best_epoch (int): Best training epoch.
        start_train (float): Training start time.
        end_train (float): Training end time.
        start_test (float): Testing start time.
        end_test (float): Testing end time.
        file_name (str): Descriptive name of the experiment.
        last_mape_values (np.ndarray): MAPE values at last timestep per sample.
    """
    results_file = os.path.join(results_dir, results_filename)

    # Define consistent column order
    columns = [
        'filename', 'train_ratio', 'smooth_speeds', 'filter_on_train_only', 'window_size',
        'P', 'Q',
        'MAE', 'MedianAE', 'RMSE', 'MAPE', 'SMAPE',
        'MAE_std', 'MedianAE_std', 'RMSE_std', 'MAPE_std','SMAPE_std',
        'learning_rate', 'total_training_time', 'validation_time_last_epoch_only',
        'inference_time', 'L', 'K', 'best_epoch', 'test_start', 'test_end']

    # Create the CSV file if it doesn't exist yet
    if not os.path.exists(results_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(results_file, index=False)

    # Construct the row with consistent ordering (use getattr to avoid missing attr errors)
    new_row = {
        'filename': args.filename,
        'train_ratio': args.train_ratio,
        'smooth_speeds': getattr(args, 'smooth_speeds', None),
        'filter_on_train_only': getattr(args, 'filter_on_train_only', None),
        'window_size': getattr(args, 'window_size', None),
        'P': args.P,
        'Q': args.Q,
        'MAE': np.mean(AE),
        'MedianAE': np.median(AE),
        'RMSE': np.sqrt(np.mean(AE ** 2)),
        'MAPE': np.mean(MAPE) * 100,
        'SMAPE': np.mean(SMAPE) * 100,
        'MAE_std': np.std(AE),
        'MedianAE_std': np.std(AE),
        'RMSE_std': np.std(AE),
        'MAPE_std': np.std(last_mape_values) * 100,
        'SMAPE_std': np.std(last_smape_values) * 100,
        'learning_rate': args.learning_rate,
        'total_training_time_all_epochs': end_train - start_train,
        "validation_time_last_epoch_only": end_val - start_val,
        'inference_time': end_test - start_test,
        'L': args.L,
        'K': args.K,
        'best_epoch': best_epoch,
        'test_start': datetime.datetime.fromtimestamp(start_test).strftime('%Y-%m-%d %H:%M:%S'),
        'test_end': datetime.datetime.fromtimestamp(end_test).strftime('%Y-%m-%d %H:%M:%S')
    }

    # Load, append, reorder, and save
    df = pd.read_csv(results_file)
    df = df.append(new_row, ignore_index=True)
    df = df[columns]  # Enforce column order
    df.to_csv(results_file, index=False)
    print(f"✅ Metrics saved to {results_file}")


class DummyLog:
    """
    A dummy log object that does nothing.
    Used when logging is disabled.
    """

    def write(self, _):
        pass

    def flush(self):
        pass


def run_gman_light(P=21, Q=15, learning_rate=0.001, L=3, K=8, max_epoch=1,
                   patience=5, batch_size=16,
                   traffic_file="data/NDW/ndw_three_weeks_gman.csv",
                   train_ratio=0.34, val_ratio=0.33, test_ratio=0.33,
                   enable_logging=False, save_results_to_csv=True,
                   results_filename='model_results.csv',
                   # New parameter to specify the directory for both outputs:
                   results_dir="results/third_experimental_study/second_gman_to_use_w_xgb", experiment_filename=None,
                   **kwargs):
    """
    Lightweight version of the GMAN model runner with automatic GPU memory release.
    Saves both the key metrics CSV file and the full predictions (Parquet) in the same directory.

    Args:
        ... (other parameters remain the same)
        results_dir (str): Directory where both CSV and Parquet files are saved.
        results_filename (str): Filename for the CSV key metrics.
        **kwargs: Additional keyword arguments to override defaults.

    Returns:
        dict: {'test_predictions': testPred, 'results': batch_results_df}
    """
    # Ensure the output directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Define default arguments in a Namespace
    args = Namespace(
        time_slot=1,
        P=P,
        Q=Q,
        L=L,
        K=K,
        d=8,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=batch_size,
        max_epoch=max_epoch,
        patience=patience,
        learning_rate=learning_rate,
        decay_epoch=5,
        traffic_file=traffic_file,
        SE_file=kwargs.get('SE_file', 'data/NDW/SE_new.txt'),
        model_file='results/GMAN(NDW)',
        log_file='results/log(NDW)',
        metrics_file='results/metrics.csv',
        filename=experiment_filename
    )

    for key, value in kwargs.items():
        setattr(args, key, value)

    # Load data and reset TensorFlow graph
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = load_data(
        args, None)
    num_train, _, N = trainX.shape
    tf.compat.v1.reset_default_graph()

    # Build the GMAN model
    X, TE, label, is_training, pred, loss, train_op, global_step = build_gman(
        args, SE, 24 * 60 // args.time_slot, N, None, num_train, mean, std
    )

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        best_epoch, start_train, end_train, end_val = train_gman(
            args, sess, X, TE, label, is_training, train_op, loss,
            trainX, trainTE, trainY, valX, valTE, valY, None
        )
        testPred, start_test, end_test, batch_results_df = test_gman_opt(
            args, sess, X, TE, is_training, pred, testX, testTE, testY, None
        )
        testPred = testPred.reshape(testY.shape)

        # Log evaluation metrics
        last_ae, last_rmse, last_mape, last_mape_values, last_smape, last_smape_values = log_metrics(
            log=None, trainPred=None, trainY=None, valPred=None, valY=None,
            testPred=testPred, testY=testY, args=args,
            start_test=start_test, end_test=end_test
        )
        print(f"Training/validation complete. Best epoch: {best_epoch}")

    # Save key metrics CSV file in the specified directory
    if save_results_to_csv:
        save_results(args, results_dir, results_filename, last_ae, last_rmse, last_mape,
                     best_epoch, start_train, end_train, end_val, start_test, end_test, args.filename, last_mape_values)

    # Save the full predictions as a Parquet file in the same directory
    parquet_filename = f"gman_results_P{args.P}_Q{args.Q}_max_epoch{args.max_epoch}_patience{args.patience}.parquet"
    parquet_file_path = os.path.join(results_dir, parquet_filename)
    batch_results_df.to_parquet(parquet_file_path)
    print(f"Predictions saved to {parquet_file_path}")

    return {'test_predictions': testPred, 'results': batch_results_df}


def run_gman_light_test_set_as_input_column(P=21, Q=15, learning_rate=0.001, L=3, K=8, max_epoch=1,
                                            patience=5, batch_size=16,
                                            traffic_file="data/NDW/ndw_three_weeks_gman.csv",
                                            df_gman=None,
                                            train_ratio=0.2,  # Only define val_ratio now
                                            enable_logging=False, save_results_to_csv=True,
                                            results_filename='model_results.csv',
                                            results_dir="results/third_experimental_study/second_gman_to_use_w_xgb",
                                            experiment_filename=None,
                                            **kwargs):
    """
    Lightweight version of the GMAN model runner with automatic GPU memory release.
    Supports manual test sets via 'test_set' column and splits remaining data into train/val.
    Saves both the key metrics CSV file and the full predictions (Parquet) in the same directory.

    Args:
        val_ratio (float): Proportion of non-test data to use as validation.
        traffic_file (str): CSV file containing time series with 'test_set' column.
        ...
    Returns:
        dict: {'test_predictions': testPred, 'results': batch_results_df}
    """
    # Ensure the output directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Analyze traffic file to compute ratios dynamically
    os.makedirs(results_dir, exist_ok=True)

    # Load DataFrame
    df = pd.read_csv(traffic_file) if traffic_file is not None else df_gman

    if 'test_set' not in df.columns:
        raise ValueError(
            "The input DataFrame must contain a 'test_set' column (boolean).")

    num_total = len(df)
    num_test = df['test_set'].sum()
    actual_test_ratio = num_test / num_total
    actual_train_ratio = train_ratio
    actual_val_ratio = 1.0 - actual_train_ratio - actual_test_ratio

    if actual_val_ratio < 0:
        raise ValueError(
            f"Invalid split: val_ratio < 0. Train ratio too high for test set size.")
    print(
        f"train_ratio: {actual_train_ratio}, val_ratio: {actual_val_ratio}, test_ratio: {actual_test_ratio}")
    # Define default arguments in a Namespace
    args = Namespace(
        time_slot=1,
        P=P,
        Q=Q,
        L=L,
        K=K,
        d=8,
        train_ratio=actual_train_ratio,
        val_ratio=actual_val_ratio,
        test_ratio=actual_test_ratio,  # still passed for logging/debug
        batch_size=batch_size,
        max_epoch=max_epoch,
        patience=patience,
        learning_rate=learning_rate,
        decay_epoch=5,
        df_gman=df_gman,
        traffic_file=traffic_file,
        SE_file=kwargs.get('SE_file', 'data/NDW/SE_new.txt'),
        model_file='results/GMAN(NDW)',
        log_file='results/log(NDW)',
        metrics_file='results/metrics.csv',
        filename=experiment_filename
    )

    for key, value in kwargs.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Load data and reset TensorFlow graph
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = utils.loadData_test_set_as_input_column(
        args, None)
    num_train, _, N = trainX.shape
    tf.compat.v1.reset_default_graph()

    # Build the GMAN model
    X, TE, label, is_training, pred, loss, train_op, global_step = build_gman(
        args, SE, 24 * 60 // args.time_slot, N, None, num_train, mean, std
    )

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        best_epoch, start_train, end_train,start_val, end_val = train_gman(
            args, sess, X, TE, label, is_training, train_op, loss,
            trainX, trainTE, trainY, valX, valTE, valY, None
        )
        testPred, start_test, end_test, batch_results_df = test_gman_opt(
            args, sess, X, TE, is_training, pred, testX, testTE, testY, None
        )
        testPred = testPred.reshape(testY.shape)

        # Log evaluation metrics
        last_ae, last_rmse, last_mape, last_mape_values, last_smape, last_smape_values = log_metrics(
            log=None, trainPred=None, trainY=None, valPred=None, valY=None,
            testPred=testPred, testY=testY, args=args,
            start_test=start_test, end_test=end_test
        )
        print(f"Training/validation complete. Best epoch: {best_epoch}")

    # Save key metrics CSV file in the specified directory
    if save_results_to_csv:
        save_results(args, results_dir, results_filename, last_ae, last_rmse, last_mape, last_smape,
                     best_epoch, start_train, end_train, start_val, end_val, start_test, end_test, args.filename, last_mape_values, last_smape_values)

    # Save the full predictions as a Parquet file in the same directory
    parquet_filename = f"gman_results_P{args.P}_Q{args.Q}_max_epoch{args.max_epoch}_patience{args.patience}.parquet"
    parquet_file_path = os.path.join(results_dir, parquet_filename)
    batch_results_df.to_parquet(parquet_file_path)
    print(f"Predictions saved to {parquet_file_path}")

    return {'test_predictions': testPred, 'results': batch_results_df}


def print_date_differences(tfd, df_gman_test):
    """
    Print date differences between XGBoost and GMAN test datasets.

    Args:
        tfd: An object with a DataFrame (df_for_ML) containing XGBoost test data.
        df_gman_test (DataFrame): GMAN test results DataFrame.
    """
    df_xgboost_all = tfd.df_for_ML
    df_xgboost_all['date'] = pd.to_datetime(df_xgboost_all['date'])
    df_gman_test['timestamp'] = pd.to_datetime(df_gman_test['timestamp'])
    df_xgboost_all_test = df_xgboost_all[df_xgboost_all['test_set'] == True]
    xgboost_test_date_start = df_xgboost_all_test['date'].min()
    xgboost_test_date_end = df_xgboost_all_test['date'].max()
    gman_test_date_start = df_gman_test['timestamp'].min()
    gman_test_date_end = df_gman_test['timestamp'].max()
    start_date_diff = (gman_test_date_start -
                       xgboost_test_date_start).seconds / 60
    end_date_diff = (gman_test_date_end - xgboost_test_date_end).seconds / 60
    print(f"Test set xgboost start date: {xgboost_test_date_start}")
    print(f"Test set gman start date: {gman_test_date_start}")
    print(f"Start date difference: {start_date_diff} min "
          f"(should be 21 mins, small discrepancies due to rounding)")
    print(f"Test set xgboost end date: {xgboost_test_date_end}")
    print(f"Test set gman end date: {gman_test_date_end}")
    print(f"End date difference: {end_date_diff} min "
          f"(should be 15 min, because gman can reach the final data point)")


def modify_gman(df_gman_test):
    """
    Prepares GMAN test results for analysis by:
    - Filtering for the final prediction timestep
    - Calculating prediction horizon based on timestamps
    - Reshaping the data to long format for per-sensor predictions

    Assumes:
    - Each prediction sample includes multiple timesteps
    - Prediction columns end with '_pred'

    Args:
        df_gman_test (pd.DataFrame): Wide-format GMAN results DataFrame. Must include
                                     'timestamp' or 'date', 'timestep', 'sample_nr', and prediction columns.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns:
                      - 'sensor_id': sensor identifier
                      - 'gman_prediction': predicted value
                      - 'target_date': date of the forecasted value
                      - 'prediction_date': date when the forecast was made
    """
    # --- Handle timestamp column ---
    if 'timestamp' in df_gman_test.columns:
        date_col = 'timestamp'
    elif 'date' in df_gman_test.columns:
        date_col = 'date'
    else:
        raise ValueError(
            "Expected 'timestamp' or 'date' column in the input DataFrame.")

    df_gman_test[date_col] = pd.to_datetime(df_gman_test[date_col])

    # --- Compute prediction horizon (based on first prediction sample) ---
    first_sample = df_gman_test[df_gman_test['sample_nr'] == 1]
    horizon = first_sample[date_col].max(
    ) - first_sample[date_col].min() + pd.Timedelta(minutes=1)

    # --- Keep only the last timestep's predictions ---
    max_timestep = df_gman_test['timestep'].max()
    df_latest = df_gman_test[df_gman_test['timestep'] == max_timestep]

    # --- Extract relevant columns ---
    pred_cols = [col for col in df_latest.columns if col.endswith('_pred')]
    df_latest = df_latest[[date_col] + pred_cols]

    # --- Rename sensor columns (strip '_pred') ---
    df_latest.columns = [date_col] + [col[:-5] for col in pred_cols]

    # --- Reshape to long format ---
    df_long = df_latest.melt(
        id_vars=date_col,
        var_name='sensor_id',
        value_name='gman_prediction'
    )

    # --- Add timestamps for target and prediction ---
    df_long.rename(columns={date_col: 'gman_target_date'}, inplace=True)
    df_long['gman_prediction_date'] = df_long['gman_target_date'] - horizon

    return df_long


def modify_gman_deprecated(df_gman_test):
    """
    Modify the GMAN results DataFrame for further analysis.
    Assumes 1min intervals between timestamps (dt)

    Args:
        df_gman_test (DataFrame): Original GMAN test results DataFrame.

    Returns:
        DataFrame: Modified DataFrame in long format with columns for timestamp,
                   sensor_id, and gman_prediction.
    """
    if 'timestamp' in df_gman_test.columns:
        date_col = 'timestamp'
    elif 'date' in df_gman_test.columns:
        date_col = 'date'
    else:
        raise ValueError(
            "No 'timestamp' or 'date' column found in the DataFrame.")

    # Convert 'date' column to datetime if it's not already
    df_gman_test[date_col] = pd.to_datetime(df_gman_test[date_col])

    # Calculate automatically the prediction horizon
    df_gman_test_sample = df_gman_test.loc[df_gman_test['sample_nr'] == 1]
    # Add 1 minute to include the last timestep (otherwise the difference would be for example 14 (if horizon is 15min))
    horizon = df_gman_test_sample[date_col].max(
    ) - df_gman_test_sample[date_col].min() + pd.Timedelta(minutes=1)
    max_timestep = df_gman_test['timestep'].max()
    # dt = df_gman_test['timestamp'].iloc[1] - df_gman_test['timestamp'].iloc[0]
    df_gman_test = df_gman_test[df_gman_test['timestep']
                                == max_timestep]

    # Select only the timestamp and sensor prediction columns (ending with '_pred')
    sensor_cols = [
        col for col in df_gman_test.columns if col.endswith('_pred')]
    cols_to_use = [date_col] + sensor_cols
    df_gman_test = df_gman_test[cols_to_use]

    # Rename columns: remove '_pred' suffix from sensor names
    # remove the last 5 characters ('_pred')
    new_sensor_cols = [col[:-5] for col in sensor_cols]
    df_gman_test.columns = [date_col] + new_sensor_cols

    # Reshape the DataFrame from wide to long format
    df_gman_test_long = df_gman_test.melt(
        id_vars=date_col,
        var_name='sensor_id',
        value_name='gman_prediction'
    )
    df_gman_test_long.rename(
        columns={date_col: 'target_date'}, inplace=True)
    df_gman_test_long['target_date'] = pd.to_datetime(
        df_gman_test_long['target_date'])
    df_gman_test_long['prediction_date'] = df_gman_test_long['target_date'] - horizon

    return df_gman_test_long


# def load_gman_results(p, q, directory="saved_gman_results"):
#     """
#     Load previously saved GMAN results from a Parquet file.

#     Args:
#         p (int): Parameter P.
#         q (int): Parameter Q.
#         directory (str): Directory where results are stored.

#     Returns:
#         DataFrame or None: Loaded results DataFrame, or None if file not found.
#     """
#     file_path = os.path.join(directory, f"gman_results_P{p}_Q{q}.parquet")
#     if os.path.exists(file_path):
#         return pd.read_parquet(file_path)
#     else:
#         print(f"File not found: {file_path}")
#         return None


def load_gman_results(p, q, directory="saved_gman_results"):
    """
    Load previously saved GMAN results from Parquet files that start with gman_results_P{p}_Q{q}.
    If there are multiple files, the first one will be selected.
    Args:
        p (int): Parameter P.
        q (int): Parameter Q.
        directory (str): Directory where results are stored. Returns:
        DataFrame or None: Loaded results DataFrame, or None if files are not found.
    """
    file_pattern = os.path.join(directory, f"gman_results_P{p}_Q{q}*.parquet")
    matching_files = glob.glob(file_pattern)
    if matching_files:
        print(f"Found files: {matching_files}")
        return pd.read_parquet(matching_files[0])  # Select the first file
    else:
        print(f"No files found matching {file_pattern}")
        return None
