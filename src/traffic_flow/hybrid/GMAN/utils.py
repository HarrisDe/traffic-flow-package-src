import numpy as np
import pandas as pd
import ast

# log string


def log_string(log, string):
    if log is not None:
        log.write(string + '\n')
        log.flush()
    print(string)

# metric


# def metric(pred, label):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = np.not_equal(label, 0)
#         mask = mask.astype(np.float32)
#         mask /= np.mean(mask)
#         mae = np.abs(np.subtract(pred, label)).astype(np.float32)
#         rmse = np.square(mae)
#         mape = np.divide(mae, label)
#         mae = np.nan_to_num(mae * mask)
#         mae = np.mean(mae)
#         rmse = np.nan_to_num(rmse * mask)
#         rmse = np.sqrt(np.mean(rmse))
#         mape = np.nan_to_num(mape * mask)
#         mape = np.mean(mape)
#     return mae, rmse, mape

def metric(pred, label, return_mean=True):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)

        rmse = np.nan_to_num(rmse * mask)

        mape = np.nan_to_num(mape * mask)

        if return_mean:
            mae = np.mean(mae)
            mape = np.mean(mape)
            rmse = np.sqrt(np.mean(rmse))
    return mae, rmse, mape



# def metric(pred, label, return_mean=True, epsilon=1e-6):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = np.not_equal(label, 0).astype(np.float32)
#         mask /= np.mean(mask)

#         mae = np.abs(np.subtract(pred, label)).astype(np.float32)
#         rmse = np.square(mae)
#         mape = np.divide(mae, label)
#         smape = 2.0 * mae / (np.abs(pred) + np.abs(label) + epsilon)

#         # Apply mask and handle NaNs
#         mae = np.nan_to_num(mae * mask)
#         rmse = np.nan_to_num(rmse * mask)
#         mape = np.nan_to_num(mape * mask)
#         smape = np.nan_to_num(smape * mask)

#         if return_mean:
#             mae = np.mean(mae)
#             rmse = np.sqrt(np.mean(rmse))
#             mape = np.mean(mape)
#             smape = np.mean(smape)

#     return mae, rmse, mape, smape


def metric_unweighted(pred, label):
    mae = np.mean(np.abs(pred - label))
    rmse = np.sqrt(np.mean(np.square(pred - label)))
    mape = np.mean(np.abs((pred - label) / label))
    return mae, rmse, mape


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    # print(f"data in seq2instance: {data}")
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]

    return x, y

# Original


def loadData(args, output_timestamps=False):
    # Traffic
    # df = pd.read_hdf(args.traffic_file)
    df = pd.read_csv(args.traffic_file, header=0,
                     index_col=0)
    df.index = pd.to_datetime(df.index)

    Traffic = df.values
    # train/val/test
    # Temporal Embedding

    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    test_time = df.index.to_numpy(dtype="datetime64[ms]").reshape(-1, 1)
    test_time = test_time[-test_steps:]
    # test_time = test_time[-test_steps +args.P + 1:]
    print(f"Time shape: {test_time.shape}")
    # Apply seq2instance to timestamps in the same way as testY
    _, timestamps_testY = seq2instance(
        test_time, args.P, args.Q)  # Extract timestamps

    # Convert timestamps to NumPy datetime format
    timestamps_testY = timestamps_testY.astype("datetime64[ms]")

    # Fix format: Convert to Pandas-style datetime string ('YYYY-MM-DD HH:MM:SS')
    # timestamps_testY = pd.Series(timestamps_testY).dt.strftime(
    #     '%Y-%m-%d %H:%M:%S').to_numpy()

    # spatial embedding
    f = open(args.SE_file, mode='r')

    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    # temporal embedding
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #            // Time.freq.delta.total_seconds()
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
        // (60 * 5)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    if output_timestamps:
        return timestamps_testY

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


# Modified
def loadData_test_set_as_input_column(args, output_timestamps=False):
    # Load CSV and parse datetime index

    if args.traffic_file is None:
        df = args.df_gman
    else:
        df = pd.read_csv(args.traffic_file, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Split off test set using 'test_set' boolean column
    test_mask = df['test_set'].astype(bool)
    df = df.drop(columns=['test_set'])

    df_test = df[test_mask]
    df_rest = df[~test_mask]

    # Split remaining data into train/val using ratios
    Traffic_test = df_test.values
    Traffic_rest = df_rest.values
    # num_step = Traffic_rest.shape[0]
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    # val_steps = num_step - train_steps
    train = Traffic_rest[:train_steps]
    val = Traffic_rest[train_steps:]
    val_steps = val.shape[0]
    print(
        f'Calculated train ratio: {round(train_steps/num_step,2)}, calculated validation ratio: {round(val_steps/num_step,2)},')

    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(Traffic_test, args.P, args.Q)

    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # Test timestamps
    test_time = df_test.index.to_numpy(dtype="datetime64[ms]").reshape(-1, 1)
    _, timestamps_testY = seq2instance(test_time, args.P, args.Q)
    timestamps_testY = timestamps_testY.astype("datetime64[ms]")
    print(f"Time shape: {test_time.shape}")
    print(
        f"first test timestamp (from timestamps_testY): {timestamps_testY[0]}")
    print(f"first test timestamp (from df): {df_test.index[0]}")
    print(f"first val timestamp (from df): {df_rest.index[train_steps]}")

    # Spatial Embedding
    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.strip().split(' ')
        index = int(temp[0])
        SE[index] = list(map(float, temp[1:]))

    # Temporal embedding
    def generate_time_features(time_index):
        dayofweek = np.reshape(time_index.weekday, newshape=(-1, 1))
        timeofday = (time_index.hour * 3600 + time_index.minute *
                     60 + time_index.second) // (60 * 5)
        timeofday = np.reshape(timeofday, newshape=(-1, 1))
        return np.concatenate((dayofweek, timeofday), axis=-1)

    train_time = generate_time_features(df_rest.index[:train_steps])
    val_time = generate_time_features(df_rest.index[train_steps:])
    test_time = generate_time_features(df_test.index)

    trainTE = seq2instance(train_time, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val_time, args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test_time, args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    if output_timestamps:
        return timestamps_testY

    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std


def convert_timestamp_data(timestamp):
    """
    Converts a timestamp from a string that looks like a list into a properly formatted datetime string.

    Args:
        timestamp (str or list or np.ndarray): The timestamp data to be converted.

    Returns:
        str: A properly formatted timestamp in 'YYYY-MM-DD HH:MM:SS' format.
    """
    # Step 1: Convert string to actual list if necessary
    if isinstance(timestamp, str) and timestamp.startswith("[") and timestamp.endswith("]"):
        timestamp = ast.literal_eval(timestamp)  # Convert string to list

    # Step 2: Extract first element if it's in a list or array
    if isinstance(timestamp, (list, np.ndarray)):
        timestamp = timestamp[0]

    # Step 3: Convert timestamp to proper datetime format
    return pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
