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
random.seed(69); np.random.seed(69); tf.random.set_seed(69)

from .GMAN import utils as utils
from .GMAN import model as model


def set_random_seed(seed: int = 42, use_gpu: bool = True, deterministic: bool = True) -> int:
    """
    Set seeds for Python, NumPy, and TensorFlow (TF 2.x).

    Args:
        seed (int): Seed to use across libs.
        use_gpu (bool): If False, disable GPUs for this process.
        deterministic (bool): If True, request deterministic kernels where possible.

    Returns:
        int: The seed that was set.

    Notes:
    - Call this *before* building models/datasets.
    - Full determinism depends on your ops/versions; some GPU ops remain nondeterministic.
    """
    import os, random
    import numpy as np
    import tensorflow as tf

    # 1) Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2) TensorFlow seed (prefer the unified helper when available)
    try:
        tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy, and TF
    except Exception:
        tf.random.set_seed(seed)

    # 3) Determinism knobs (best-effort)
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        try:
            # Available in TF ≥ 2.9
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass

    # 4) GPU selection / memory behavior
    if not use_gpu:
        # Must be set *before* TF initializes the runtime; still try to hide GPUs at runtime too.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("GPU disabled for reproducibility (CUDA_VISIBLE_DEVICES = -1).")
    else:
        # Enable memory growth to reduce OOM flakiness (keeps determinism reasonable).
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                print(f"GPU enabled ({len(gpus)} found). Note: some ops may still be non-deterministic.")
        except Exception:
            pass

    return seed


# --- TF2 niceties ---
def _enable_gpu_memory_growth():
    import tensorflow as tf
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


class _ValTimerCallback(tf.keras.callbacks.Callback):
    """Capture last epoch's validation timing during model.fit(..., validation_data=...)."""
    def __init__(self):
        super().__init__()
        import time
        self._time = time
        self.last_val_start = None
        self.last_val_end = None

    # Keras triggers test-begin/-end hooks for validation inside fit
    def on_test_begin(self, logs=None):
        self.last_val_start = self._time.time()

    def on_test_end(self, logs=None):
        self.last_val_end = self._time.time()
        


class BNMomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, steps_per_epoch, decay_epoch, initial=0.5, decay_rate=0.5):
        super().__init__()
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.decay_epoch = max(1, int(decay_epoch))
        self.initial = float(initial)
        self.decay_rate = float(decay_rate)
        self.global_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        # TF1 formula:
        # bn_momentum = initial * (decay_rate ** floor(global_step / decay_steps))
        decay_steps = self.decay_epoch * self.steps_per_epoch
        power = self.global_step // max(1, decay_steps)
        bn_momentum = self.initial * (self.decay_rate ** power)
        bn_decay = min(0.99, 1.0 - bn_momentum)  # Keras BN "momentum"

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = bn_decay

def build_gman_old(args, SE, T, N, log, num_train, mean, std):
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


class ClippedExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate, staircase=True, min_lr=1e-5):
        self._base = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
        self._min_lr = tf.constant(min_lr, dtype=tf.float32)
    def __call__(self, step):
        return tf.maximum(self._base(step), self._min_lr)

    def get_config(self):
        # Optional: makes it serializable if you ever save/restore via config
        return {
            "initial_lr": self._base.initial_learning_rate,
            "decay_steps": self._base.decay_steps,
            "decay_rate": self._base.decay_rate,
            "staircase": self._base.staircase,
            "min_lr": float(self._min_lr.numpy()),
        }

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
    from .GMAN import model as gman_model  # local import to avoid circulars

    # --- Keras inputs (replace placeholders) ---
    X_in  = tf.keras.Input(shape=(args.P, N), dtype=tf.float32, name="X")
    TE_in = tf.keras.Input(shape=(args.P + args.Q, 2), dtype=tf.int32, name="TE")

    # --- Build GMAN body (pure tensor ops) ---
    # model.GMAN already returns a Tensor given (X, TE, SE, ...)
    pred = gman_model.GMAN(
        X_in, TE_in, tf.constant(SE, dtype=tf.float32),
        args.P, args.Q, T, args.L, args.K, args.d,
        bn=True, bn_decay=0.9, is_training=None  # is_training is ignored in TF2 path
    )

    # Rescale predictions back to original space
    pred = pred * std + mean

    # --- Wrap into a tf.keras.Model ---
    net = tf.keras.Model(inputs=[X_in, TE_in], outputs=pred, name="GMAN")

    # --- LR schedule & compile ---
    steps_per_epoch = max(1, num_train // max(1, args.batch_size))
    lr_schedule = ClippedExpDecay(
        initial_lr=args.learning_rate,
        decay_steps=args.decay_epoch * steps_per_epoch,
        decay_rate=0.7,
        staircase=True,
        min_lr=1e-5,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         epsilon=1e-8,         # <-- match TF1 default
                                         )

    # Use your existing masked last-step MAE as a Keras loss
    net.compile(optimizer=optimizer, loss=gman_model.last_step_masked_mae_keras)

    if log is not None:
        from .GMAN import utils
        utils.log_string(log, "model compiled (TF2/Keras)!")
    return net


def train_gman_old(args, sess, X, TE, label, is_training, train_op, loss,
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


def train_gman(args, net, trainX, trainTE, trainY, valX, valTE, valY, log):
    """
    TF2/Keras training with:
    - EarlyStopping (patience=args.patience, restore best)
    - ModelCheckpoint (best .keras)
    - Validation timing captured via callback

    Returns:
        best_epoch, start_train, end_train, start_val, end_val
    """
    import os, time, numpy as np, tensorflow as tf

    
    steps_per_epoch = max(1, trainX.shape[0] // max(1, args.batch_size))
    bn_sched = BNMomentumScheduler(
        steps_per_epoch=steps_per_epoch,
        decay_epoch=args.decay_epoch,
        initial=0.5, decay_rate=0.5
    )
    
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.log_dir or ".", "gman_best.keras"),
            monitor="val_loss", save_best_only=True
        ),
        bn_sched,
        _ValTimerCallback()
    ]

    start_train = time.time()
    hist = net.fit(
        x=[trainX, trainTE], y=trainY,
        validation_data=([valX, valTE], valY),
        epochs=args.max_epoch, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1, shuffle=True,
    )
    # extract timing from the ValTimer that was in callbacks
    val_timer = next((cb for cb in callbacks if isinstance(cb, _ValTimerCallback)), None)
    start_val = val_timer.last_val_start if val_timer and val_timer.last_val_start is not None else time.time()
    end_val   = val_timer.last_val_end   if val_timer and val_timer.last_val_end   is not None else start_val
    end_train = start_val

    # Best epoch (1-based)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

    if log is not None:
        from .GMAN import utils
        utils.log_string(log, f"Training complete. Best epoch: {best_epoch}")
    return best_epoch, start_train, end_train, start_val, end_val




def test_gman_opt_old(args, sess, X, TE, is_training, pred, testX, testTE, testY, log):
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

def test_gman_opt(args, net, testX, testTE, testY, log=None):
    """
    TF2/Keras test:
    Returns:
        testPred: np.ndarray of shape (num_test, Q, N)
        start_test: float (epoch seconds)
        end_test:   float
        batch_results_df: pd.DataFrame same schema as _old version:
          ['batch_nr','sample_nr','timestep','timestamp', <sensor>, <sensor>_pred, ...]
    """
    import time, numpy as np, pandas as pd
    from . import utils as _u  # already imported above as "import utils", but safe here

    num_test = testX.shape[0]
    num_timesteps = testY.shape[1]
    N = testY.shape[2]

    # Resolve sensor ids
    if getattr(args, "traffic_file", None):
        try:
            sensor_ids = pd.read_csv(args.traffic_file, index_col=0).columns.tolist()
        except Exception:
            sensor_ids = [f"S{i}" for i in range(N)]
    elif getattr(args, "df_gman", None) is not None:
        try:
            sensor_ids = args.df_gman.drop("test_set", axis=1).columns.tolist()
        except Exception:
            sensor_ids = [f"S{i}" for i in range(N)]
    else:
        sensor_ids = [f"S{i}" for i in range(N)]

    # Timestamps for test set (shape to (num_test, num_timesteps))
    ts = _u.loadData_test_set_as_input_column(args, output_timestamps=True)
    # Robustly coerce to ndarray
    if isinstance(ts, tuple):
        ts = ts[0]
    timestamps_testY = np.array(ts).reshape(num_test, num_timesteps)

    # Predict
    start_test = time.time()
    testPred = net.predict([testX, testTE], batch_size=args.batch_size, verbose=0)
    end_test = time.time()

    # Build per-step long dataframe like the _old implementation
    pred_flat   = testPred.reshape(-1, N)   # (num_test*num_timesteps, N)
    actual_flat = testY.reshape(-1, N)
    timestamps_flat = timestamps_testY.reshape(-1)

    # Convert to human-readable timestamps
    convert_ts = np.vectorize(_u.convert_timestamp_data)
    timestamps_converted = convert_ts(timestamps_flat)

    # Batch numbers (1-based), per *sample* then repeated across timesteps
    sample_idx = np.arange(num_test)
    batch_nr_per_sample = (sample_idx // max(1, args.batch_size)) + 1
    batch_nrs_flat = np.repeat(batch_nr_per_sample, num_timesteps)

    data_dict = {
        "batch_nr": batch_nrs_flat,
        "sample_nr": np.repeat(np.arange(1, num_test + 1), num_timesteps),
        "timestep": np.tile(np.arange(1, num_timesteps + 1), num_test),
        "timestamp": timestamps_converted,
    }
    for s, name in enumerate(sensor_ids):
        data_dict[name] = actual_flat[:, s]
        data_dict[f"{name}_pred"] = pred_flat[:, s]

    batch_results_df = pd.DataFrame(data_dict)

    return testPred, start_test, end_test, batch_results_df


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
    'filename','train_ratio','smooth_speeds','filter_on_train_only','window_size',
    'P','Q',
    'MAE','MedianAE','RMSE','MAPE','SMAPE',
    'MAE_std','MedianAE_std','RMSE_std','MAPE_std','SMAPE_std',
    'learning_rate','total_training_time_all_epochs','validation_time_last_epoch_only',
    'inference_time','L','K','best_epoch','test_start','test_end'
]

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



# def run_gman_light_test_set_as_input_column(P=21, Q=15, learning_rate=0.001, L=3, K=8, max_epoch=1,
#                                             patience=5, batch_size=16,
#                                             traffic_file="data/NDW/ndw_three_weeks_gman.csv",
#                                             df_gman=None,
#                                             train_ratio=0.2,  # Only define val_ratio now
#                                             enable_logging=False, save_results_to_csv=True,
#                                             results_filename='model_results.csv',
#                                             results_dir="results/third_experimental_study/second_gman_to_use_w_xgb",
#                                             experiment_filename=None,
#                                             **kwargs):
#     """
#     Lightweight version of the GMAN model runner with automatic GPU memory release.
#     Supports manual test sets via 'test_set' column and splits remaining data into train/val.
#     Saves both the key metrics CSV file and the full predictions (Parquet) in the same directory.

#     Args:
#         val_ratio (float): Proportion of non-test data to use as validation.
#         traffic_file (str): CSV file containing time series with 'test_set' column.
#         ...
#     Returns:
#         dict: {'test_predictions': testPred, 'results': batch_results_df}
#     """
#     # Ensure the output directory exists
#     os.makedirs(results_dir, exist_ok=True)

#     # Analyze traffic file to compute ratios dynamically
#     os.makedirs(results_dir, exist_ok=True)

#     # Load DataFrame
#     df = pd.read_csv(traffic_file) if traffic_file is not None else df_gman

#     if 'test_set' not in df.columns:
#         raise ValueError(
#             "The input DataFrame must contain a 'test_set' column (boolean).")

#     num_total = len(df)
#     num_test = df['test_set'].sum()
#     actual_test_ratio = num_test / num_total
#     actual_train_ratio = train_ratio
#     actual_val_ratio = 1.0 - actual_train_ratio - actual_test_ratio

#     if actual_val_ratio < 0:
#         raise ValueError(
#             f"Invalid split: val_ratio < 0. Train ratio too high for test set size.")
#     print(
#         f"train_ratio: {actual_train_ratio}, val_ratio: {actual_val_ratio}, test_ratio: {actual_test_ratio}")
#     # Define default arguments in a Namespace
#     args = Namespace(
#         time_slot=1,
#         P=P,
#         Q=Q,
#         L=L,
#         K=K,
#         d=8,
#         train_ratio=actual_train_ratio,
#         val_ratio=actual_val_ratio,
#         test_ratio=actual_test_ratio,  # still passed for logging/debug
#         batch_size=batch_size,
#         max_epoch=max_epoch,
#         patience=patience,
#         learning_rate=learning_rate,
#         decay_epoch=5,
#         df_gman=df_gman,
#         traffic_file=traffic_file,
#         SE_file=kwargs.get('SE_file', 'data/NDW/SE_new.txt'),
#         model_file='results/GMAN(NDW)',
#         log_file='results/log(NDW)',
#         metrics_file='results/metrics.csv',
#         filename=experiment_filename
#     )

#     for key, value in kwargs.items():
#         if not hasattr(args, key):
#             setattr(args, key, value)

#     # Load data and reset TensorFlow graph
#     trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = utils.loadData_test_set_as_input_column(
#         args, None)
#     num_train, _, N = trainX.shape


#     # Build the GMAN model
#     X, TE, label, is_training, pred, loss, train_op, global_step = build_gman(
#         args, SE, 24 * 60 // args.time_slot, N, None, num_train, mean, std
#     )


#     sess.run(tf.compat.v1.global_variables_initializer())
#         best_epoch, start_train, end_train,start_val, end_val = train_gman(
#             args, sess, X, TE, label, is_training, train_op, loss,
#             trainX, trainTE, trainY, valX, valTE, valY, None
#         )
#         testPred, start_test, end_test, batch_results_df = test_gman_opt(
#             args, sess, X, TE, is_training, pred, testX, testTE, testY, None
#         )
#         testPred = testPred.reshape(testY.shape)

#         # Log evaluation metrics
#         last_ae, last_rmse, last_mape, last_mape_values, last_smape, last_smape_values = log_metrics(
#             log=None, trainPred=None, trainY=None, valPred=None, valY=None,
#             testPred=testPred, testY=testY, args=args,
#             start_test=start_test, end_test=end_test
#         )
#         print(f"Training/validation complete. Best epoch: {best_epoch}")

#     # Save key metrics CSV file in the specified directory
#     if save_results_to_csv:
#         save_results(args, results_dir, results_filename, last_ae, last_rmse, last_mape, last_smape,
#                      best_epoch, start_train, end_train, start_val, end_val, start_test, end_test, args.filename, last_mape_values, last_smape_values)

#     # Save the full predictions as a Parquet file in the same directory
#     parquet_filename = f"gman_results_P{args.P}_Q{args.Q}_max_epoch{args.max_epoch}_patience{args.patience}.parquet"
#     parquet_file_path = os.path.join(results_dir, parquet_filename)
#     batch_results_df.to_parquet(parquet_file_path)
#     print(f"Predictions saved to {parquet_file_path}")

#     return {'test_predictions': testPred, 'results': batch_results_df}

def _sanitize_gman_df(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent: keep DatetimeIndex, numeric sensors, and 'test_set' bool."""
    df = df.copy()
    # Promote datetime column to index if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("timestamp", "date", "datetime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    if 'test_set' not in df.columns:
        raise ValueError("Expected 'test_set' column in df.")
    if df['test_set'].dtype != bool:
        df['test_set'] = df['test_set'].astype(bool)
    keep = ['test_set'] + [c for c in df.columns if c != 'test_set' and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep]


def run_gman_light_test_set_as_input_column(
    P=21, Q=15, learning_rate=0.001, L=3, K=8, max_epoch=1,
    patience=5, batch_size=16,
    traffic_file="data/NDW/ndw_three_weeks_gman.csv",
    df_gman=None,
    train_ratio=0.2,  # Only define val_ratio now; test set comes from 'test_set'
    save_results_to_csv=True,
    results_filename='model_results.csv',
    results_dir="results/third_experimental_study/second_gman_to_use_w_xgb",
    experiment_filename=None,
    **kwargs
):
    """
    TF2/Keras-compatible driver.
    - Uses utils.loadData_test_set_as_input_column(...) for splits.
    - Builds Keras model via build_gman(...)
    - Trains with train_gman(...)
    - Tests with test_gman_opt(...) and returns the same batch_results_df shape as old version
    - Writes CSV + parquet like before
    """
    import os, pandas as pd, numpy as np, tensorflow as tf
    os.makedirs(results_dir, exist_ok=True)
    _enable_gpu_memory_growth()

    # Load DataFrame if path given, else use provided df_gman
    df = pd.read_csv(traffic_file) if traffic_file is not None else df_gman
    if 'test_set' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'test_set' column (boolean).")
    
    df = _sanitize_gman_df(df)
    num_total = len(df)
    num_test = df['test_set'].sum()
    actual_test_ratio  = num_test / num_total
    actual_train_ratio = train_ratio
    actual_val_ratio   = 1.0 - actual_train_ratio - actual_test_ratio
    if actual_val_ratio < 0:
        raise ValueError("Invalid split: val_ratio < 0. Train ratio too high for test set size.")

    # Build arg namespace (keeps downstream utils untouched)
    args = Namespace(
        time_slot=1,
        P=P, Q=Q,
        L=L, K=K, d=8,
        train_ratio=actual_train_ratio,
        val_ratio=actual_val_ratio,
        test_ratio=actual_test_ratio,
        batch_size=batch_size,
        max_epoch=max_epoch,
        patience=patience,
        learning_rate=learning_rate,
        decay_epoch=5,
        df_gman=df,
        traffic_file=traffic_file,
        SE_file=kwargs.get('SE_file', 'data/NDW/SE_new.txt'),
        model_file=os.path.join(results_dir, 'GMAN(NDW)'),
        log_file=os.path.join(results_dir, 'log(NDW)'),
        metrics_file=os.path.join(results_dir, 'metrics.csv'),
        filename=experiment_filename,
        log_dir=results_dir,
    )
    for key, value in kwargs.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Load arrays + scalers
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = utils.loadData_test_set_as_input_column(
        args, None
    )
    num_train, _, N = trainX.shape

    # Build compiled Keras model
    net = build_gman(args, SE, 24 * 60 // args.time_slot, N, None, num_train, mean, std)

    # Train (captures start_val/end_val for CSV)
    best_epoch, start_train, end_train, start_val, end_val = train_gman(
        args, net, trainX, trainTE, trainY, valX, valTE, valY, log=None
    )

    # Test (also returns batch_results_df)
    testPred, start_test, end_test, batch_results_df = test_gman_opt(
        args, net, testX, testTE, testY, log=None
    )

    # Compute + log metrics (unchanged)
    last_ae, last_rmse, last_mape, last_mape_values, last_smape, last_smape_values = log_metrics(
        log=None, trainPred=None, trainY=None, valPred=None, valY=None,
        testPred=testPred, testY=testY, args=args,
        start_test=start_test, end_test=end_test
    )
    print(f"Training/validation complete. Best epoch: {best_epoch}")

    # Save metrics CSV
    if save_results_to_csv:
        save_results(
            args, results_dir, results_filename,
            last_ae, last_rmse, last_mape, last_smape,
            best_epoch, start_train, end_train, start_val, end_val, start_test, end_test,
            args.filename, last_mape_values, last_smape_values
        )

    # Save full predictions parquet (same name pattern)
    parquet_filename = f"gman_results_P{args.P}_Q{args.Q}_max_epoch{args.max_epoch}_patience{args.patience}.parquet"
    parquet_file_path = os.path.join(results_dir, parquet_filename)
    batch_results_df.to_parquet(parquet_file_path)
    print(f"Predictions saved to {parquet_file_path}")

    return {"test_predictions": testPred, "results": batch_results_df}


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
