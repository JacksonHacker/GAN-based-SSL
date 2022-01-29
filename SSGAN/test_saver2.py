import tensorflow as tf

tf.reset_default_graph()

v1 = tf.get_variable('v1', shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable('v2', shape=[5], initializer = tf.zeros_initializer)

# We'll only restore part of the parameters
# You can also save the specific parameters in this way.
saver = tf.train.Saver({'v2': v2})

with tf.Session() as sess:
    # You need to initialize other parameters,
    # because they will not be restored.
    v1.initializer.run()
    saver.restore(sess, './checkpoints/model.ckpt')

    print('v1: %s' % v1.eval())
    print('v2: %s' % v2.eval())