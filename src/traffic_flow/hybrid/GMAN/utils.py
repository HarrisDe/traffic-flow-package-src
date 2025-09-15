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
    df = pd.read_csv(args.traffic_file, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)

    Traffic = df.values

    # splits
    num_step   = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps  = round(args.test_ratio  * num_step)
    val_steps   = num_step - train_steps - test_steps

    train = Traffic[: train_steps]
    val   = Traffic[train_steps: train_steps + val_steps]
    test  = Traffic[-test_steps:]

    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX,   valY   = seq2instance(val,   args.P, args.Q)
    testX,  testY  = seq2instance(test,  args.P, args.Q)

    # normalization (fit on train only)
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX   = (valX   - mean) / std
    testX  = (testX  - mean) / std

    # --- timestamps for testY (for downstream reporting) ---
    test_time = df.index.to_numpy(dtype="datetime64[ms]").reshape(-1, 1)
    test_time = test_time[-test_steps:]
    _, timestamps_testY = seq2instance(test_time, args.P, args.Q)
    timestamps_testY = timestamps_testY.astype("datetime64[ms]")
    if output_timestamps:
        return timestamps_testY

    # --- spatial embedding ---
    with open(args.SE_file, "r") as f:
        lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros((N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    # --- temporal embedding (DAYOFWEEK, TIMEOFDAY) ---
    # time_slot in minutes; default to 1 if not provided
    time_slot = int(getattr(args, "time_slot", 1))
    T = 24 * 60 // max(1, time_slot)

    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))  # 0..6

    # Minutes-of-day binned by time_slot
    # (equivalent to floor((h*60 + m) / time_slot))
    mod = (Time.hour * 60 + Time.minute) // time_slot
    # Ensure range 0..T-1 in case of off-by-one/frequency quirks
    timeofday = (mod % T).astype(np.int64).reshape(-1, 1)

    # Combine -> shape [num_step, 2]
    Time_enc = np.concatenate((dayofweek, timeofday), axis=-1)

    # Split TE same as data
    trainTE = seq2instance(Time_enc[: train_steps], args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)  # (num_train, P+Q, 2)

    valTE = seq2instance(Time_enc[train_steps: train_steps + val_steps], args.P, args.Q)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)

    testTE = seq2instance(Time_enc[-test_steps:], args.P, args.Q)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    # Safety checks (helps catch mismatches with STEmbedding one-hot depth)
    assert trainTE[...,0].min() >= 0 and trainTE[...,0].max() <= 6
    assert valTE[...,0].min()   >= 0 and valTE[...,0].max()   <= 6
    assert testTE[...,0].min()  >= 0 and testTE[...,0].max()  <= 6
    assert trainTE[...,1].min() >= 0 and trainTE[...,1].max() <  T
    assert valTE[...,1].min()   >= 0 and valTE[...,1].max()   <  T
    assert testTE[...,1].min()  >= 0 and testTE[...,1].max()  <  T

    return (trainX, trainTE, trainY, valX, valTE, valY,
            testX,  testTE,  testY,  SE,  mean,  std)


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
    num_step = df.shape[0]  # keep your current choice
    train_steps = round(args.train_ratio * num_step)
    train = Traffic_rest[:train_steps]
    val   = Traffic_rest[train_steps:]
    val_steps = val.shape[0]
    print(f'Calculated train ratio: {round(train_steps/num_step,2)}, calculated validation ratio: {round(val_steps/num_step,2)},')

    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX,   valY   = seq2instance(val,   args.P, args.Q)
    testX,  testY  = seq2instance(Traffic_test, args.P, args.Q)

    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX   = (valX   - mean) / std
    testX  = (testX  - mean) / std

    # Test timestamps (for logging/optional return)
    test_time_idx = df_test.index.to_numpy(dtype="datetime64[ms]").reshape(-1, 1)
    _, timestamps_testY = seq2instance(test_time_idx, args.P, args.Q)
    timestamps_testY = timestamps_testY.astype("datetime64[ms]")
    print(f"Time shape: {test_time_idx.shape}")
    print(f"first test timestamp (from timestamps_testY): {timestamps_testY[0]}")
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

    # -------- Temporal embedding (UPDATED) --------
    time_slot = int(getattr(args, "time_slot", 1))
    T = 24 * 60 // max(1, time_slot)

    def _to_numpy_1d(x):
        return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)

    def generate_time_features(time_index: pd.DatetimeIndex) -> np.ndarray:
        # day of week: 0..6
        dow = _to_numpy_1d(time_index.weekday).astype(np.int64).reshape(-1, 1)
        # minutes since midnight
        minutes = _to_numpy_1d(time_index.hour) * 60 + _to_numpy_1d(time_index.minute)
        # time of day bucket 0..T-1
        tod = _to_numpy_1d((minutes // time_slot) % T).astype(np.int64).reshape(-1, 1)
        return np.concatenate((dow, tod), axis=-1)

    train_time_feat = generate_time_features(df_rest.index[:train_steps])
    val_time_feat   = generate_time_features(df_rest.index[train_steps:])
    test_time_feat  = generate_time_features(df_test.index)

    trainTE = np.concatenate(seq2instance(train_time_feat, args.P, args.Q), axis=1).astype(np.int32)
    valTE   = np.concatenate(seq2instance(val_time_feat,   args.P, args.Q), axis=1).astype(np.int32)
    testTE  = np.concatenate(seq2instance(test_time_feat,  args.P, args.Q), axis=1).astype(np.int32)

    # Optional checks
    assert 0 <= trainTE[...,0].min() <= 6 and 0 <= trainTE[...,0].max() <= 6
    assert 0 <=  valTE[...,0].min() <= 6 and 0 <=  valTE[...,0].max() <= 6
    assert 0 <= testTE[...,0].min() <= 6 and 0 <= testTE[...,0].max() <= 6
    assert 0 <= trainTE[...,1].min() <  T and 0 <= trainTE[...,1].max() <  T
    assert 0 <=  valTE[...,1].min() <  T and 0 <=  valTE[...,1].max() <  T
    assert 0 <= testTE[...,1].min() <  T and 0 <= testTE[...,1].max() <  T

    if output_timestamps:
        return timestamps_testY

    return (trainX, trainTE, trainY,
            valX,   valTE,  valY,
            testX,  testTE, testY,
            SE, mean, std)

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
