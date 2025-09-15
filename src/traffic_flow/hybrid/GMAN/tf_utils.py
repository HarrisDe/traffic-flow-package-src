# import tensorflow as tf

# #import tensorflow.compat.v1 as tf
# #tf.disable_v2_behavior()

# def conv2d(x, output_dims, kernel_size, stride = [1, 1],
#            padding = 'SAME', use_bias = True, activation = tf.nn.relu,
#            bn = False, bn_decay = None, is_training = None):
#     input_dims = x.get_shape()[-1]
#     kernel_shape = kernel_size + [input_dims, output_dims]
    
#     #kernel = tf.Variable(
#     #    tf.glorot_uniform_initializer()(shape = kernel_shape),
#     #    dtype = tf.float32, trainable = True, name = 'kernel')
#     initializer = tf.keras.initializers.GlorotUniform()
#     kernel = tf.Variable(initializer(shape=kernel_shape), dtype=tf.float32, trainable=True, name='kernel')
#     x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding = padding)
#     if use_bias:
#         bias = tf.Variable(
#             tf.zeros_initializer()(shape = [output_dims]),
#             dtype = tf.float32, trainable = True, name = 'bias')
#         x = tf.nn.bias_add(x, bias)
#     if activation is not None:
#         if bn:
#             x = batch_norm(x, is_training = is_training, bn_decay = bn_decay)
#         x = activation(x)
#     return x

# def batch_norm(x, is_training, bn_decay):
#     input_dims = x.get_shape()[-1]
#     moment_dims = list(range(len(x.get_shape()) - 1))
#     beta = tf.Variable(
#         tf.zeros_initializer()(shape = [input_dims]),
#         dtype = tf.float32, trainable = True, name = 'beta')
#     gamma = tf.Variable(
#         tf.ones_initializer()(shape = [input_dims]),
#         dtype = tf.float32, trainable = True, name = 'gamma')
#     batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

#     decay = bn_decay if bn_decay is not None else 0.9
    
#     '''ema = tf.train.ExponentialMovingAverage(decay = decay)
#     # Operator that maintains moving averages of variables.
#     ema_apply_op = tf.cond(
#         is_training,
#         lambda: ema.apply([batch_mean, batch_var]),
#         lambda: tf.no_op())
#     # Update moving average and return current batch's avg and var.
#     def mean_var_with_update():
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#     # ema.average returns the Variable holding the average of var.
#     mean, var = tf.cond(
#         is_training,
#         mean_var_with_update,
#         lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
#     x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)'''

#     batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=decay, epsilon=1e-3)
#     x = batch_norm_layer(x, training=False)
#     return x

# def dropout(x, drop, is_training):
#     x = tf.cond(
#         is_training,
#         lambda: tf.nn.dropout(x, rate = drop),
#         lambda: x)
#     return x


import tensorflow as tf

# def conv2d(x, output_dims, kernel_size, stride=[1, 1],
#            padding='SAME', use_bias=True, activation=tf.nn.relu,
#            bn=False, bn_decay=None, is_training=None):
#     """
#     TF2/Keras conv2d that mirrors the old TF1 helper:
#     - same signature (callers unchanged)
#     - Keras layers underneath
#     - ignores `is_training` (Keras handles train/infer automatically)
#     """
#     # sanitize shapes & padding
#     ks = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
#     st = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride)
#     pad = 'same' if str(padding).upper() == 'SAME' else 'valid'

#     # if BN follows, bias is typically redundant
#     use_bias_eff = use_bias and not bn

#     y = tf.keras.layers.Conv2D(
#         filters=output_dims,
#         kernel_size=ks,
#         strides=st,
#         padding=pad,
#         use_bias=use_bias_eff,
#         activation=None,
#         kernel_initializer='glorot_uniform',
#     )(x)

#     if bn:
#         # Map TF1 bn_decay (~moving avg "decay") to Keras BatchNorm "momentum"
#         # Clamp to a safe range; BNMomentumScheduler will update this during training.
#         mom = 0.99 if bn_decay is None else float(max(0.0, min(0.999, bn_decay)))
#         y = tf.keras.layers.BatchNormalization(momentum=mom, epsilon=1e-3)(y)

#     if activation is not None:
#         y = tf.keras.layers.Activation(activation)(y)

#     return y

def conv2d(x, output_dims, kernel_size, stride=[1, 1],
           padding='SAME', use_bias=True, activation=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    """
    TF2/Keras conv2d that mirrors the TF1 order: Conv -> (BN) -> Activation.
    """
    strides = (int(stride[0]), int(stride[1]))
    pad = 'same' if str(padding).upper() == 'SAME' else 'valid'

    y = tf.keras.layers.Conv2D(
        filters=int(output_dims),
        kernel_size=tuple(int(k) for k in kernel_size),
        strides=strides,
        padding=pad,
        use_bias=use_bias,
        activation=None,
        kernel_initializer='glorot_uniform',
    )(x)

    if bn:
        momentum = 0.99 if bn_decay is None else float(bn_decay)
        # Let Keras set training=True/False as appropriate during fit()/predict()
        y = tf.keras.layers.BatchNormalization(
            momentum=momentum, epsilon=1e-3
        )(y)

    if activation is not None:
        y = tf.keras.layers.Activation(activation)(y)

    return y

def batch_norm(x, is_training=None, bn_decay=None):
    """
    If you still call this anywhere, map to Keras BN.
    """
    momentum = 0.99 if bn_decay is None else float(bn_decay)
    return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=1e-3)(x)


def dropout(x, drop, is_training=None):
    """
    Keras Dropout; training flag is inferred automatically by Keras.
    """
    return tf.keras.layers.Dropout(rate=drop)(x)
