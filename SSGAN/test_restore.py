import tensorflow as tf

tf.reset_default_graph()

v1 = tf.get_variable('v1', shape=[3])
v2 = tf.get_variable('v2', shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './checkpoints/model_init.ckpt')
    print('Model restored.')

    print("v1: %s" % v1.eval())
    print("v2: %s" % v2.eval())

    saver.restore(sess, './checkpoints/model.ckpt')
    print('Model restored.')

    print("v1: %s" % v1.eval())
    print("v2: %s" % v2.eval())