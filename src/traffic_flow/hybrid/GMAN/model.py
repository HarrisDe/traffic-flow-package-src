from . import tf_utils
import tensorflow as tf
import keras.ops as K


def placeholder(P, Q, N):
    X = tf.compat.v1.placeholder(shape=(None, P, N), dtype=tf.float32)
    TE = tf.compat.v1.placeholder(shape=(None, P + Q, 2), dtype=tf.int32)
    label = tf.compat.v1.placeholder(shape=(None, Q, N), dtype=tf.float32)
    is_training = tf.compat.v1.placeholder(shape=(), dtype=tf.bool)
    return X, TE, label, is_training


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        x = tf_utils.conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    # Use K.expand_dims to be Keras3-friendly with KerasTensors/constants
    SE = K.expand_dims(K.expand_dims(SE, axis=0), axis=0)
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # temporal embedding
    # tf.one_hot on a KerasTensor is fine; we keep it for exact behavior.
    dayofweek = tf.one_hot(TE[..., 0], depth=7)
    timeofday = tf.one_hot(TE[..., 1], depth=T)
    TE = K.concat((dayofweek, timeofday), axis=-1)
    TE = K.expand_dims(TE, axis=2)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return K.add(SE, TE)


# ---- Helpers to split/merge attention heads without python-side shape math ----
def _split_heads_reshape(x, K_heads, d):
    """
    Input:  x shape (B, S, N, K_heads*d)
    Return: shape (B*K_heads, S, N, d)
    All transforms use graph ops; never multiply symbolic dims in Python.
    """
    # (B, S, N, K*d) -> (B, S, N, K, d)
    x = K.reshape(x, (-1, K.shape(x)[1], K.shape(x)[2], K_heads, d))
    # (B, K, S, N, d)
    x = K.transpose(x, (0, 3, 1, 2, 4))
    # (B*K, S, N, d)
    x = K.reshape(x, (-1, K.shape(x)[2], K.shape(x)[3], K.shape(x)[4]))
    return x


def _merge_heads_concat(x, K_heads):
    """
    Inverse of _split_heads_reshape.
    Input:  x shape (B*K_heads, S, N, d)
    Return: shape (B, S, N, K_heads*d)
    """
    BK = K.shape(x)[0]
    S  = K.shape(x)[1]
    Nn = K.shape(x)[2]
    d  = K.shape(x)[3]
    # Recover B via integer division in the graph
    B = BK // K_heads
    # (B*K, S, N, d) -> (B, K, S, N, d)
    x = K.reshape(x, (B, K_heads, S, Nn, d))
    # (B, S, N, K, d) -> (B, S, N, K*d)
    x = K.transpose(x, (0, 2, 3, 1, 4))
    x = K.reshape(x, (B, S, Nn, K_heads * d))
    return x



def spatialAttention(X, STE, K_heads, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = int(K_heads * d)
    X = K.concat((X, STE), axis=-1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    # Split heads via reshape (no Python math on symbolic dims)
    # [K * batch_size, num_step, N, d]
    query = _split_heads_reshape(query, K_heads, d)
    key   = _split_heads_reshape(key,   K_heads, d)
    value = _split_heads_reshape(value, K_heads, d)

    # [K * batch_size, num_step, N, N]
    attention = K.matmul(query, K.transpose(key, (0, 1, 3, 2)))
    attention = attention / (d ** 0.5)
    attention = K.softmax(attention, axis=-1)

    # [batch_size, num_step, N, D]
    X = K.matmul(attention, value)  # (B*K, S, N, d)
    X = _merge_heads_concat(X, K_heads)  # (B, S, N, K*d)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K_heads, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K_heads * d
    Xcat = K.concat((X, STE), axis=-1)

    query = FC(
        Xcat, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        Xcat, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        Xcat, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    # [K * batch_size, num_step, N, d]
    query = _split_heads_reshape(query, K_heads, d)
    key   = _split_heads_reshape(key,   K_heads, d)
    value = _split_heads_reshape(value, K_heads, d)

    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = K.transpose(query, (0, 2, 1, 3))
    key   = K.transpose(key,   (0, 2, 3, 1))
    value = K.transpose(value, (0, 2, 1, 3))

    # [K * batch_size, N, num_step, num_step]
    attention = K.matmul(query, key)
    attention = attention / (d ** 0.5)

    # mask attention score
    if mask:
        # S = number of steps (last dim of attention)
        S  = K.shape(attention)[-1]
        KB = K.shape(attention)[0]   # K*B
        Nn = K.shape(attention)[1]   # N

        # Build (S, S) lower-triangular matrix with dynamic shape
        ones_S   = tf.ones(tf.stack([S, S]), dtype=attention.dtype)
        lower    = tf.linalg.LinearOperatorLowerTriangular(ones_S).to_dense()  # (S, S)
        lower    = K.expand_dims(K.expand_dims(lower, axis=0), axis=0)         # (1,1,S,S)

        # Tile to (K*B, N, S, S) with dynamic multiples
        multiples = tf.stack([KB, Nn, tf.constant(1, tf.int32), tf.constant(1, tf.int32)])
        lower     = tf.tile(lower, multiples)

        very_neg  = tf.cast(-2 ** 15 + 1, attention.dtype)
        attention = tf.where(lower > 0, attention, very_neg)

    # softmax
    attention = K.softmax(attention, axis=-1)

    # [batch_size, num_step, N, D]
    Xout = K.matmul(attention, value)            # (B*K, N, S, d)
    Xout = K.transpose(Xout, (0, 2, 1, 3))       # (B*K, S, N, d)
    Xout = _merge_heads_concat(Xout, K_heads)    # (B, S, N, K*d)
    Xout = FC(
        Xout, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return Xout


def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(K.add(XS, XT))
    H = K.add(K.multiply(z, HS), K.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H


def STAttBlock(X, STE, K_heads, d, bn, bn_decay, is_training, mask=False):
    HS = spatialAttention(X, STE, K_heads, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K_heads, d, bn, bn_decay, is_training, mask=mask)
    H = gatedFusion(HS, HT, K_heads * d, bn, bn_decay, is_training)
    return K.add(X, H)


def transformAttention(X, STE_P, STE_Q, K_heads, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K_heads * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        STE_P, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    # [K * batch_size, Q, N, d] ; [K * batch_size, P, N, d]
    query = _split_heads_reshape(query, K_heads, d)
    key   = _split_heads_reshape(key,   K_heads, d)
    value = _split_heads_reshape(value, K_heads, d)

    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = K.transpose(query, (0, 2, 1, 3))
    key   = K.transpose(key,   (0, 2, 3, 1))
    value = K.transpose(value, (0, 2, 1, 3))

    # [K * batch_size, N, Q, P]
    attention = K.matmul(query, key)
    attention = attention / (d ** 0.5)
    attention = K.softmax(attention, axis=-1)

    # [batch_size, Q, N, D]
    X = K.matmul(attention, value)        # (B*K, N, Q, d)
    X = K.transpose(X, (0, 2, 1, 3))      # (B*K, Q, N, d)
    X = _merge_heads_concat(X, K_heads)   # (B, Q, N, K*d)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def GMAN(X, TE, SE, P, Q, T, L, K_heads, d, bn, bn_decay, is_training):
    '''
    GMAN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    D = K_heads * d
    # input
    X = K.expand_dims(X, axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P:]
    # encoder
    for _ in range(L):
        X = STAttBlock(X, STE_P, K_heads, d, bn, bn_decay, is_training)
    # transAtt
    X = transformAttention(
        X, STE_P, STE_Q, K_heads, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K_heads, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.squeeze(X, axis=3)

# def mae_loss(pred, label):
#     """
#     Original function
#     """
#     mask = tf.not_equal(label, 0)
#     mask = tf.cast(mask, tf.float32)
#     mask /= tf.reduce_mean(mask)
#     mask = tf.compat.v2.where(
#         condition = tf.math.is_nan(mask), x = 0., y = mask)
#     loss = tf.abs(tf.subtract(pred, label))
#     loss *= mask
#     loss = tf.compat.v2.where(
#         condition = tf.math.is_nan(loss), x = 0., y = loss)
#     loss = tf.reduce_mean(loss)
#     return loss


# def mae_loss(pred, label):
#     """
#     Mean Absolute Error (MAE) loss, but only for the last predicted timestep (Qth).
#     """
#     mask = tf.not_equal(label[:, -1, :], 0)  # Only consider Qth timestep
#     mask = tf.cast(mask, tf.float32)
#     mask /= tf.reduce_mean(mask)
#     mask = tf.compat.v2.where(
#         condition=tf.math.is_nan(mask), x=0., y=mask)

#     # Compute loss **only for last timestep**
#     # Only last timestep
#     loss = tf.abs(tf.subtract(pred[:, -1, :], label[:, -1, :]))
#     loss *= mask
#     loss = tf.compat.v2.where(
#         condition=tf.math.is_nan(loss), x=0., y=loss)

#     loss = tf.reduce_mean(loss)  # Compute final MAE over batch
#     return loss


def mae_loss(pred, label):
    """
    Compute Mean Absolute Error (MAE) loss only for the last prediction timestep.
    """
    # Extract the last timestep only (Q-th step)
    pred_last = pred[:, -1, :]  # Shape: (batch_size, num_sensors)
    label_last = label[:, -1, :]  # Shape: (batch_size, num_sensors)

    # Compute absolute error only on last timestep
    loss = tf.abs(tf.subtract(pred_last, label_last))

    # Ignore zero values (optional)
    mask = tf.not_equal(label_last, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask

    # Remove NaNs and compute mean loss
    loss = tf.compat.v2.where(tf.math.is_nan(loss), x=0., y=loss)
    loss = tf.reduce_mean(loss)  # Only averages the last timestep's loss

    return loss


def last_step_masked_mae_keras(y_true, y_pred):
    """
    Wrap your existing mae_loss(pred, label) into a Keras loss signature.
    Your mae_loss computes MAE only on the last predicted step and masks zeros.
    """
    return mae_loss(pred=y_pred, label=y_true)
