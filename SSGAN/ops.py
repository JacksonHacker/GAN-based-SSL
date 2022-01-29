import tensorflow as tf

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)

def deconv2d(input, filters, kernel_size=5, strides=2, padding='same', name='deconv2d', leak=0.2, is_train=True, batch_norm=True, activation_fn=None, is_NCHW=False):
    with tf.variable_scope(name):
        if is_NCHW:
            output = tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding, data_format='channels_first')
        else:
            output = tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding, data_format='channels_last')

        if batch_norm:
            if is_NCHW:
                output = tf.layers.batch_normalization(output, axis=1, training=is_train)
            else:
                output = tf.layers.batch_normalization(output, axis=-1, training=is_train)


        if not activation_fn:
            output = lrelu(output, leak)
        else:
            ouput = activation_fn(output)

        return output

def conv2d(input, filters, kernel_size=3, strides=2, padding='same', name='conv2d', leak=0.2, is_train=True, batch_norm=True, is_NCHW=False):
    with tf.variable_scope(name):
        if is_NCHW:
            output = tf.layers.conv2d(input, filters, kernel_size, strides, padding, data_format='channels_first')
        else:
            output = tf.layers.conv2d(input, filters, kernel_size, strides, padding, data_format='channels_last')

        if batch_norm:
            if is_NCHW:
                output = tf.layers.batch_normalization(output, axis=1, training=is_train)
            else:
                output = tf.layers.batch_normalization(output, axis=-1, training=is_train)


        output = lrelu(output, leak)

        return output