import tensorflow as tf
from ops import *

def classifier_inputs(size):
    input = tf.placeholder(tf.float32, (None, *size), name='input')
    y = tf.placeholder(tf.int32, (None), name='label')

    return input, y

def classifier(x, reuse=False, leak=0.2, drop_rate=0., num_classes=10, size_mult=64, is_NCHW=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.dropout(x, rate=drop_rate / 2.5)

        # Input layer is 96 * 128 * 3


        x1 = conv2d(x,
                    size_mult,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='conv2d_1',
                    leak=leak,
                    is_train=True,
                    batch_norm=False,
                    is_NCHW=is_NCHW)
        # No Batch Norm !
        x1 = tf.layers.dropout(x1, rate=drop_rate)
        # 48 * 64 * (size_mult)

        x2 = conv2d(x1,
                    size_mult * 2,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='conv2d_2',
                    leak=leak,
                    is_train=True,
                    batch_norm=True,
                    is_NCHW=is_NCHW)
        # 24 * 32 * (size_mult * 2)

        x3 = conv2d(x2,
                    size_mult * 4,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='conv2d_3',
                    leak=leak,
                    is_train=True,
                    batch_norm=True,
                    is_NCHW=is_NCHW)
        x3 = tf.layers.dropout(x3, rate=drop_rate)
        # 12 * 16 * (size_mult * 4)


        x4 = conv2d(x3,
                    size_mult * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name='conv2d_4',
                    leak=leak,
                    is_train=True,
                    batch_norm=True,
                    is_NCHW=is_NCHW)
        # 12 * 16 * (size_mult * 4)

        x5 = conv2d(x4,
                    size_mult * 4,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name='conv2d_5',
                    leak=leak,
                    is_train=True,
                    batch_norm=True,
                    is_NCHW=is_NCHW)
        x5 = tf.layers.dropout(x5, rate=drop_rate)
        # 12 * 16 * (size_mult * 4)

        x6 = conv2d(x5,
                    size_mult * 8,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='conv2d_6',
                    leak=leak,
                    is_train=True,
                    batch_norm=True,
                    is_NCHW=is_NCHW)
        # 6 * 8 * (size_mult * 8)

        x7 = conv2d(x6,
                    size_mult * 8,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='conv2d_7',
                    leak=leak,
                    batch_norm=False,
                    is_NCHW=is_NCHW)
        # 3 * 4 * (size_mult * 8)

        # Don't use bn on this layer, because bn would set the mean of each feature
        # to the bn mu parameter.

        # Flatten it by global average pooling
        if is_NCHW:
            features = tf.reduce_mean(x7, (2, 3))
        else:
            features = tf.reduce_mean(x7, (1, 2))
        # 1 * 1 * 512

        # Set class_logits to be the inputs to a softmax distribution over the different classes
        class_logits = tf.layers.dense(features, num_classes)

        out = tf.nn.softmax(class_logits)

        return out, class_logits

def classifier_loss(input, y, num_classes, leak=0.2, drop_rate=0., is_NCHW=False):
    size_mult = 64

    prob, logits = classifier(input, leak=leak, drop_rate=drop_rate, size_mult=size_mult, is_NCHW=is_NCHW)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                               labels=tf.one_hot(y, num_classes, dtype=tf.float32))
    cross_entropy = tf.squeeze(cross_entropy)

    pred_class = tf.cast(tf.argmax(logits, 1), tf.int32)
    eq = tf.equal(tf.squeeze(y), pred_class) # a Tensor of bool
    correct = tf.reduce_sum(tf.to_float(eq))

    return cross_entropy, correct

def classifier_opt(loss, learning_rate, beta1, global_step):
    t_vars = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=t_vars, global_step=global_step)

    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

    return train_opt, shrink_lr

class vanilla_classifier:
    def __init__(self, size, learning_rate, num_classes=10, leak=0.2, beta1=0.5, is_NCHW=False):
        tf.reset_default_graph()

        self.learning_rate = tf.get_variable('learning_rate',
                                             shape=[],
                                             trainable=False,
                                             initializer=tf.constant_initializer([learning_rate]))

        self.input, self.y = classifier_inputs(size)
        self.drop_rate = tf.placeholder_with_default(.5, (), 'drop_rate')
        self.global_step = tf.train.get_or_create_global_step()

        self.loss, self.correct = classifier_loss(self.input, self.y, num_classes,
                               leak=leak,
                               drop_rate=self.drop_rate,
                               is_NCHW=is_NCHW)

        self.train_opt, self.shrink_lr = classifier_opt(self.loss, self.learning_rate, beta1, self.global_step)




