import tensorflow as tf
from ops import deconv2d, conv2d, lrelu

def model_inputs(real_dim, z_dim):
    # real_dim = (3, 96, 128) or (96, 128, 3)
    # z_dim = 100

    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    y = tf.placeholder(tf.int32, (None), name='y')
    label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

    return inputs_real, inputs_z, y, label_mask


def generator(z, output_dim, reuse=False, leak=0.2, training=True, size_mult=128, is_NCHW=False):
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, 3 * 4 * size_mult * 8)
        # Reshape it to start the convolutional stack
        if is_NCHW:
            x1 = tf.reshape(x1, (-1, size_mult * 8, 3, 4))
            x1 = tf.layers.batch_normalization(x1, axis=1, training=training)
        else:
            x1 = tf.reshape(x1, (-1, 3, 4, size_mult * 8))
            x1 = tf.layers.batch_normalization(x1, axis=-1, training=training)


        x1 = lrelu(x1, leak)
        # 3 * 4 * (size_mult * 8)

        x2 = deconv2d(x1,
                      filters=size_mult * 4,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      name='deconv2d_1',
                      leak=leak,
                      is_train=training,
                      is_NCHW=is_NCHW)
        # 6 * 8 * (size_mult * 4)

        x3 = deconv2d(x2,
                      filters=size_mult * 2,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      name='deconv2d_2',
                      leak=leak,
                      is_train=training,
                      is_NCHW=is_NCHW)
        # 12 * 16 * (size_mult * 2)

        x4 = deconv2d(x3,
                      filters=size_mult,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      name='deconv2d_3',
                      leak=leak,
                      is_train=training,
                      is_NCHW=is_NCHW)
        # 24 * 32 * (size_mult)

        x5 = deconv2d(x4,
                      filters=size_mult,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      name='deconv2d_4',
                      leak=leak,
                      is_train=training,
                      is_NCHW=is_NCHW)
        # 48 * 64 * (size_mult)

        x6 = deconv2d(x5,
                      filters=size_mult,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      name='deconv2d_5',
                      leak=leak,
                      is_train=training,
                      is_NCHW=is_NCHW)
        # 96 * 128 * (size_mult)


        # Output layer
        out = deconv2d(x6,
                       filters=output_dim,
                       kernel_size=5,
                       strides=1,
                       padding='same',
                       name='deconv2d_6',
                       is_train=training,
                       is_NCHW=is_NCHW,
                       batch_norm=False,          # No Batch Norm !
                       activation_fn=tf.tanh)
        # 96 * 128 * 3

        return out


def discriminator(x, reuse=False, leak=0.2, drop_rate=0., num_classes=10, size_mult=64, extra_class=1, is_NCHW=False):
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
        # This layer is used for the feature matching loss, which only works if
        # the means can be different when the discriminator is run on the data than
        # when the discriminator is run on the generator samples.

        # Flatten it by global average pooling
        if is_NCHW:
            features = tf.reduce_mean(x7, (2, 3))
        else:
            features = tf.reduce_mean(x7, (1, 2))
        # 1 * 1 * 512

        # Set class_logits to be the inputs to a softmax distribution over the different classes
        class_logits = tf.layers.dense(features, num_classes + extra_class)

        # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).
        # Keep in mind that class_logits gives you the probability distribution over all the real
        # classes and the fake class. You need to work out how to transform this multiclass softmax
        # distribution into a binary real-vs-fake decision that can be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)

        if extra_class:
            real_class_logits, fake_class_logits = tf.split(class_logits, [num_classes, 1], 1)
            assert fake_class_logits.get_shape()[1] == 1, fake_class_logits.get_shape()
            fake_class_logits = tf.squeeze(fake_class_logits)
        else:
            real_class_logits = class_logits
            fake_class_logits = 0.

        mx = tf.reduce_max(real_class_logits, 1, keepdims=True)
        stable_real_class_logits = real_class_logits - mx

        gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_real_class_logits), 1)) + tf.squeeze(mx) - fake_class_logits

        out = tf.nn.softmax(class_logits)

        return out, class_logits, gan_logits, features


def model_loss(input_real, input_z, output_dim, y, num_classes, label_mask, leak=0.2, drop_rate=0., extra_class=1, is_NCHW=False):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param output_dim: The number of channels in the output image
    :param y: Integer class labels
    :param num_classes: The number of classes
    :param alpha: The slope of the left half of leaky ReLU activation
    :param drop_rate: The probability of dropping a hidden unit
    :return: A tuple of (discriminator loss, generator loss)
    """

    # These numbers multiply the size of each layer of the generator and the discriminator,
    # respectively. You can reduce them to run your code faster for debugging purposes.
    g_size_mult = 64
    d_size_mult = 64

    # Here we run the generator and the discriminator
    g_model = generator(input_z, output_dim, leak=leak, size_mult=g_size_mult, is_NCHW=is_NCHW)

    # The default arg 'reuse' is False.
    d_on_data = discriminator(input_real, leak=leak, drop_rate=drop_rate, size_mult=d_size_mult, extra_class=extra_class, is_NCHW=is_NCHW)
    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data

    # Don't forget to set 'reuse = True'.
    d_on_samples = discriminator(g_model, reuse=True, leak=leak, drop_rate=drop_rate, size_mult=d_size_mult, extra_class=extra_class, is_NCHW=is_NCHW)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples

    # Here we compute `d_loss`, the loss for the discriminator.
    # This should combine two different losses:
    #  1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
    #     real-vs-fake classification problem.
    #  2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
    #     for the multi-class softmax. For this one we use the labels. Don't forget to ignore
    #     use `label_mask` to ignore the examples that we are pretending are unlabeled for the
    #     semi-supervised learning problem.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data, labels=tf.ones_like(gan_logits_on_data)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples, labels=tf.zeros_like(gan_logits_on_samples)))
    y = tf.squeeze(y)
    class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=class_logits_on_data,
                                                                  labels=tf.one_hot(y, num_classes + extra_class,
                                                                                    dtype=tf.float32))
    class_cross_entropy = tf.squeeze(class_cross_entropy)
    label_mask = tf.squeeze(tf.to_float(label_mask))
    d_loss_class = tf.reduce_sum(label_mask * class_cross_entropy) / tf.maximum(1., tf.reduce_sum(label_mask))
    d_loss = d_loss_class + d_loss_real + d_loss_fake

    # Here we set `g_loss` to the "feature matching" loss invented by Tim Salimans at OpenAI.
    # This loss consists of minimizing the absolute difference between the expected features
    # on the data and the expected features on the generated samples.
    # This loss works better for semi-supervised learning than the tradition GAN losses.
    data_moments = tf.reduce_mean(data_features, axis=0)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
    eq = tf.equal(tf.squeeze(y), pred_class)
    correct = tf.reduce_sum(tf.to_float(eq))
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

    return d_loss, g_loss, correct, masked_correct, g_model


def model_opt(d_loss, g_loss, learning_rate, beta1, global_step):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and biases to update. Get them separately for the discriminator and the generator
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    for t in t_vars:
        assert t in d_vars or t in g_vars

    # Minimize both players' costs simultaneously
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars, global_step=global_step)
    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

    return d_train_opt, g_train_opt, shrink_lr


class GAN:
    """
    A GAN model.
    :param real_size: The shape of the real data.
    :param z_size: The number of entries in the z code vector.
    :param learnin_rate: The learning rate to use for Adam.
    :param num_classes: The number of classes to recognize.
    :param alpha: The slope of the left half of the leaky ReLU activation
    :param beta1: The beta1 parameter for Adam.
    """

    def __init__(self, real_size, z_size, learning_rate, num_classes=10, leak=0.2, beta1=0.5, extra_class=1, is_NCHW=False):
        # real_size = (96,128,3)
        # z_size = 100
        # learning_rate = 0.0003

        tf.reset_default_graph()

        self.learning_rate = tf.get_variable('learning_rate',
                                             shape=[],
                                             trainable=False,
                                             initializer=tf.constant_initializer([learning_rate]))
        self.input_real, self.input_z, self.y, self.label_mask = model_inputs(real_size, z_size)
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
        self.global_step = tf.train.get_or_create_global_step()

        if is_NCHW:
            output_channels = real_size[0]
        else:
            output_channels = real_size[-1]

        loss_results = model_loss(self.input_real, self.input_z,
                                  output_channels, self.y, num_classes, label_mask=self.label_mask,
                                  leak=leak,
                                  drop_rate=self.drop_rate, extra_class=extra_class, is_NCHW=is_NCHW)
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples = loss_results

        self.d_opt, self.g_opt, self.shrink_lr = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1, self.global_step)
