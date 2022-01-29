import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import time
from config import every_steps_to_save, ckpt_dir, generated_dir
import os


def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes

def train(net,
          dataset_train,
          dataset_test,
          training_init_op,
          test_init_op,
          next_element,
          epochs,
          batch_size,
          z_size,
          figsize=(5, 5), is_restore=False):
    saver_init = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=3) # Saver is a class
    saver_restore = tf.train.Saver()
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    samples, train_accuracies, test_accuracies = [], [], []

    # config=tf.ConfigProto(log_device_placement=True)
    with tf.Session() as sess:
        if not is_restore:
            sess.run(tf.global_variables_initializer())
            save_path = saver_init.save(sess, os.path.join(ckpt_dir, 'ssgan_init'))
            print('Model saved in path: %s' % save_path)
        else:
            saver_restore.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            print('Model restored from path: %s' % tf.train.latest_checkpoint(ckpt_dir))

        for e in range(epochs):
            print("Epoch", e)


            num_examples = 0
            num_correct = 0

            sess.run(training_init_op)
            idx = 0

            train_start = time.time()
            while True:
                try:
                    x, y, label_mask  = sess.run(next_element)
                    assert 'int' in str(y.dtype)
                    print('data format when training: {}'.format(x.shape))
                    num_examples += label_mask.sum()

                    batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                    # Run optimizers
                    t1 = time.time()
                    _, _, correct, global_step = sess.run([net.d_opt, net.g_opt, net.masked_correct, net.global_step],
                                             feed_dict={net.input_real: x, net.input_z: batch_z,
                                                        net.y: y, net.label_mask: label_mask})

                    t2 = time.time()
                    num_correct += correct

                    if (idx + 1) % every_steps_to_save == 0:
                        save_path = saver.save(sess, os.path.join(ckpt_dir, 'ssgan'), global_step=global_step)
                        print('Model saved in path: %s' % save_path)

                    if idx + 1 == 2:
                        break

                    idx += 1
                except tf.errors.OutOfRangeError:
                    break

            train_end = time.time()
            sess.run([net.shrink_lr])

            train_accuracy = num_correct / float(num_examples)

            print("\t\tClassifier train accuracy: ", train_accuracy)


            ############################################################################


            num_examples = 0
            num_correct = 0

            sess.run(test_init_op)
            test_start = time.time()
            while True:
                try:
                    x, y, _ = sess.run(next_element)
                    assert 'int' in str(y.dtype)
                    num_examples += x.shape[0]

                    correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                                  net.y: y,
                                                                  net.drop_rate: 0.})
                    num_correct += correct
                except tf.errors.OutOfRangeError:
                    break
            test_end = time.time()

            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy: ", test_accuracy)
            print("\t\tSingle minibatch time: ", t2 - t1)
            print("\t\tTrain time: ", train_end - train_start)
            print("\t\tTest time: ", test_end - test_start)

            gen_samples = sess.run(
                net.samples,
                feed_dict={net.input_z: sample_z})
            samples.append(gen_samples)
            _ = view_samples(-1, samples, 5, 10, figsize=figsize)
            steps = net.global_step.eval()
            plt.savefig(os.path.join(generated_dir, 'global step ' + str(steps)))
            print('save generated images.')
            # plt.show()

            # Save history of accuracies to view after training
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        global_step = net.global_step.eval()
        save_path = saver.save(sess, os.path.join(ckpt_dir, 'final'), global_step=global_step)
        print('Model saved in path: %s' % save_path)

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return train_accuracies, test_accuracies, samples