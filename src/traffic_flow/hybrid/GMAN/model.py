from . import tf_utils
import tensorflow as tf
import keras.ops as K
from tensorflow.keras.layers import Lambda, Concatenate, Softmax
from tensorflow.keras.layers import Add, Multiply, Activation
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


# def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
#     """
#     spatio-temporal embedding
#     SE: [N, D]
#     TE: [batch, P+Q, 2] (dayofweek, timeofday)
#     T:  num of time steps in one day
#     D:  output dims
#     return: [batch, P+Q, N, D]
#     """
#     # spatial embedding
#     SE = K.expand_dims(K.expand_dims(SE, axis=0), axis=0)
#     SE = FC(SE, units=[D, D], activations=[tf.nn.relu, None],
#             bn=bn, bn_decay=bn_decay, is_training=is_training)

#     # temporal embedding (use keras.ops.one_hot to stay Keras3-safe)
#     dayofweek = tf.one_hot(TE[..., 0], 7)
#     timeofday = tf.one_hot(TE[..., 1], T)
#     TE = tf.concat((dayofweek, timeofday), axis=-1)
#     TE = tf.expand_dims(TE, axis=2)
#     TE = FC(TE, units=[D, D], activations=[tf.nn.relu, None],
#             bn=bn, bn_decay=bn_decay, is_training=is_training)
#     return K.add(SE, TE)


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
    SE = K.expand_dims(K.expand_dims(SE, axis=0), axis=0)
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    # temporal embedding
    day_idx = Lambda(lambda t: t)(TE[..., 0])
    dayofweek = Lambda(
        lambda t: tf.one_hot(tf.cast(t, tf.int32), depth=7),
        output_shape=lambda s: (s[0], s[1], 7),
        dtype="float32",
    )(day_idx)

    tod_idx = Lambda(lambda t: t)(TE[..., 1])
    timeofday = Lambda(
        lambda t, depth=T: tf.one_hot(tf.cast(t, tf.int32), depth=depth),
        output_shape=lambda s, depth=T: (s[0], s[1], depth),
        dtype="float32",
    )(tod_idx)

    # <-- Replace tf.concat with the layer variant
    TE = Concatenate(axis=-1)([dayofweek, timeofday])

    TE = Lambda(
        lambda t: tf.expand_dims(t, axis=2),
        output_shape=lambda s: (s[0], s[1], 1, s[2]),
    )(TE)

    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    return K.add(SE, TE)


# ---- Helpers to split/merge attention heads without python-side shape math ----
def _split_heads_reshape(x, K_heads, d):
    """(B,S,N,K*d) -> (B*K, S, N, d) by folding heads into batch."""
    def fn(t):
        shape = tf.shape(t)        # [B, S, N, K*d]
        B, S, N = shape[0], shape[1], shape[2]
        return tf.reshape(t, (B * K_heads, S, N, d))
    # Let Keras infer output shape; if you want to specify:
    # return Lambda(fn, output_shape=lambda s: (None, s[1], s[2], d), name="split_heads")(x)
    return Lambda(fn)(x)

def _merge_heads_concat(x, K_heads):
    """(B*K, S, N, d) -> (B, S, N, K*d) by unfolding heads out of batch and concatenating."""
    def fn(t):
        shape = tf.shape(t)        # [B*K, S, N, d]
        BK, S, N, d = shape[0], shape[1], shape[2], shape[3]
        B = BK // K_heads
        t = tf.reshape(t, (B, K_heads, S, N, d))   # (B, K, S, N, d)
        t = tf.transpose(t, (0, 2, 3, 1, 4))       # (B, S, N, K, d)
        return tf.reshape(t, (B, S, N, K_heads * d))
    # Let Keras infer output shape; if you want to specify:
    # return Lambda(fn, output_shape=lambda s: (None, s[1], s[2], None), name="merge_heads")(x)
    return Lambda(fn)(x)



def spatialAttention(X, STE, K_heads, d, bn, bn_decay, is_training):
    """
    X, STE: (B, S, N, D)   -> returns (B, S, N, D) with D = K_heads * d
    """
    D = int(K_heads * d)

    # Keep a 4D reference for batch size recovery during merge:
    X_ref = X

    # concat features for Q/K/V
    inp = Concatenate(axis=-1)([X, STE])

    query = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    key   = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)

    q = _split_heads_reshape(query, K_heads, d)   # (B*K, S, N, d)
    k = _split_heads_reshape(key,   K_heads, d)   # (B*K, S, N, d)
    v = _split_heads_reshape(value, K_heads, d)   # (B*K, S, N, d)

    # attention = tf.matmul(q, k, transpose_b=True)  # (B*K, S, N, N)
    att = Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True))([q, k])

    att = Lambda(lambda a: a / tf.sqrt(tf.cast(d, a.dtype)))(att)
    att = Softmax(axis=-1)(att)

    # x = tf.matmul(att, v)  -> (B*K, S, N, d)
    x = Lambda(lambda t: tf.matmul(t[0], t[1]))([att, v])


    X = _merge_heads_concat(x, K_heads)  # (B, S, N, K*d)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K_heads, d, bn, bn_decay, is_training, mask=True):
    D = int(K_heads * d)

    # BEFORE: inp = tf.concat((X, STE), axis=-1)
    inp = Concatenate(axis=-1)([X, STE])

    query = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    key   = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(inp, units=D, activations=tf.nn.relu, bn=bn, bn_decay=bn_decay, is_training=is_training)

    q = _split_heads_reshape(query, K_heads, d)   # (B*K, S, N, d)
    k = _split_heads_reshape(key,   K_heads, d)
    v = _split_heads_reshape(value, K_heads, d)

    # Transpose to (B*K, N, S, d) / (B*K, N, d, S) / (B*K, N, S, d)
    qT = Lambda(lambda t: tf.transpose(t, (0, 2, 1, 3)))(q)
    kT = Lambda(lambda t: tf.transpose(t, (0, 2, 3, 1)))(k)
    vT = Lambda(lambda t: tf.transpose(t, (0, 2, 1, 3)))(v)

    att = Lambda(lambda t: tf.matmul(t[0], t[1]))([qT, kT])

    if mask:
        def mask_fn(a):
            BK = tf.shape(a)[0]; N = tf.shape(a)[1]; S = tf.shape(a)[2]
            m  = tf.linalg.band_part(tf.ones((S, S), dtype=a.dtype), -1, 0)  # (S,S)
            m  = tf.reshape(m, (1, 1, S, S))
            m  = tf.tile(m, (BK, N, 1, 1))                                   # (B*K, N, S, S)
            neg = tf.cast(-(2**15 - 1), a.dtype)
            return tf.where(m > 0, a, neg)
        att = Lambda(mask_fn)(att)

    att = Lambda(lambda a: a / tf.sqrt(tf.cast(d, a.dtype)))(att)
    att = Softmax(axis=-1)(att)

    xN = Lambda(lambda t: tf.matmul(t[0], t[1]))([att, vT])  # (B*K, N, S, d)
    xS = Lambda(lambda t: tf.transpose(t, (0, 2, 1, 3)))(xN)  # (B*K, S, N, d)

    X  = _merge_heads_concat(xS, K_heads)  # (B, S, N, K*d) # (B, S, N, K*d)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None], bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X



def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    XS = FC(HS, units=D, activations=None, bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=False)
    XT = FC(HT, units=D, activations=None, bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=True)

    # ✅ correct chaining: Add() produces a tensor, then Activation consumes it
    sum_x = Add()([XS, XT])
    z = Activation('sigmoid')(sum_x)

    one_minus_z = Lambda(lambda t: 1.0 - t)(z)
    H = Add()([
        Multiply()([z, HS]),
        Multiply()([one_minus_z, HT]),
    ])

    H = FC(H, units=[D, D], activations=[tf.nn.relu, None],
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
    #X = K.expand_dims(X, axis=-1)
    X = Lambda(lambda t: tf.expand_dims(t, axis=-1))(X)
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
    X = Lambda(lambda t: tf.squeeze(t, axis=3))(X)
    #return K.squeeze(X, axis=3)
    return X

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
    loss = K.abs(K.subtract(pred_last, label_last))

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
