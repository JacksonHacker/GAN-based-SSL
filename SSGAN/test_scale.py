import tensorflow as tf
import numpy as np


def scale(x, sess, feature_range=(-1, 1)):

    x = tf.cast(x, tf.float32)
    print('After cast to tf.float32: \n{}'.format(sess.run(x)))
    x = tf.truediv(x, 255.)
    print('After truediv: \n{}'.format(sess.run(x)))
    x = tf.multiply(x, 2.)
    print('After multiply: \n{}'.format(sess.run(x)))
    x = x - 1.
    print('Finally, you get: \n{}'.format(sess.run(x)))
    return x

w = tf.Variable([[ 87,  58, 187],
                 [ 30, 198,  43],
                 [241, 235,  55]], dtype=tf.uint8 , name='input_w')
print(w)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # np_w = sess.run(w)
    # np_w = np.ceil(np_w)
    # np_w = np_w.astype(int)
    # print(type(np_w))
    # print(np_w)
    print("w: \n{}".format(sess.run(w)))
    scale(w, sess)