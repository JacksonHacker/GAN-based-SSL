import tensorflow as tf
import time
import os
from classifier_config import every_steps_to_save, ckpt_dir

def train(net,
          dataset_train,
          dataset_test,
          training_init_op,
          test_init_op,
          next_element,
          epochs,
          is_restore=False):
    saver = tf.train.Saver(max_to_keep=4)
    saver_restore = tf.train.Saver()

    train_accuracies, test_accuracies = [], []

    with tf.Session() as sess:
        if is_restore:
            saver_restore.restore(sess, 'checkpoints\ssgan-4')
            print('Model restored from path: %s' % 'checkpoints\ssgan-4')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialization.')

        for e in range(epochs):
            print("Epoch", e)

            t1e = time.time()
            num_examples = 0
            num_correct = 0

            sess.run(training_init_op)
            idx = 0

            train_start = time.time()
            while True:
                try:
                    x, y = sess.run(next_element)
                    assert 'int' in str(y.dtype)
                    num_examples += x.shape[0]

                    t1 = time.time()
                    _, correct, global_step = sess.run([net.train_opt, net.correct, net.global_step],
                                                       feed_dict={net.input: x, net.y: y})
                    t2 = time.time()
                    num_correct += correct

                    if (idx + 1) % every_steps_to_save == 0:
                        save_path = saver.save(sess,
                                               os.path.join(ckpt_dir, 'classifier'),
                                               global_step=global_step)
                        print('Model saved in path: %s' % save_path)

                    # if idx + 1 == 2:
                    #     break

                    idx += 1

                except tf.errors.OutOfRangeError:
                    break
            train_end = time.time()
            sess.run([net.shrink_lr])

            train_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier train accuracy: ", train_accuracy)


            #########################################################################


            num_examples = 0
            num_correct = 0

            sess.run(test_init_op)
            test_start = time.time()
            while True:
                try:
                    x, y = sess.run(next_element)
                    assert 'int' in str(y.dtype)
                    num_examples += x.shape[0]

                    correct, = sess.run([net.correct],
                                       feed_dict={net.input: x,
                                                  net.y: y,
                                                  net.drop_rate: 0.})
                    num_correct += correct
                except tf.errors.OutOfRangeError:
                    break
            test_end = time.time()


            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tSingle minibatch time: ", t2 - t1)
            print("\t\tTrain time: ", train_end - train_start)
            print("\t\tTest time: ", test_end - test_start)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        global_step = net.global_step.eval()
        save_path = saver.save(sess, os.path.join(ckpt_dir, 'final'), global_step=global_step)
        print('Model saved in path: %s' % save_path)

    return train_accuracies, test_accuracies


