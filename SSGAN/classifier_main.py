import os
import matplotlib.pyplot as plt
import tensorflow as tf
from classifier import vanilla_classifier
from classifier_train import train
from os.path import isdir
from classifier_config import *



net = vanilla_classifier(input_size, learning_rate, is_NCHW=is_NCHW)


########################################################################
with tf.device("/cpu:0"):
    def scale(x, feature_range=(-1,1)):
        x = tf.cast(x, tf.float32)
        x = tf.truediv(x, 255.)
        x = tf.multiply(x, 2.)
        x = x - 1.
        return x

def parse(serialized):
    features = \
        {
            'image/encoded':   tf.FixedLenFeature([], tf.string),
            'image/label':     tf.FixedLenFeature([], tf.int64)
        }

    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    image_raw = parsed_example['image/encoded']
    image_unit8 = tf.image.decode_jpeg(image_raw, channels=3)
    label = parsed_example['image/label']


    if is_NCHW:
        image_unit8 = tf.transpose(image_unit8, [2, 0, 1])
    print(image_unit8)
    print("image format: {0}".format(image_unit8.get_shape()))
    return scale(image_unit8), label

def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


if not isdir(data_dir):
    raise Exception("Data directory doesn't exist!")

train_filename_list = [_get_dataset_filename(data_dir, "train", i, tfrecord_filename, train_num_shards) for i in range(train_num_shards)]
test_filename_list = [_get_dataset_filename(data_dir, "test", i, tfrecord_filename, test_num_shards) for i in range(test_num_shards)]
dataset_train = tf.data.TFRecordDataset(train_filename_list)
dataset_test = tf.data.TFRecordDataset(test_filename_list)
dataset_train = dataset_train.map(parse)
dataset_test = dataset_test.map(parse)


dataset_train.shuffle(buffer_size=buffer_size)
dataset_train = dataset_train.batch(batch_size)
dataset_test = dataset_test.batch(batch_size)

iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                           dataset_train.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(dataset_train)
test_init_op = iterator.make_initializer(dataset_test)


train_accuracies, test_accuracies = train(net,
                                          dataset_train,
                                          dataset_test,
                                          training_init_op,
                                          test_init_op,
                                          next_element,
                                          epochs,
                                          is_restore=False)

accuracies_filename = os.path.join(statistics_dir, 'classifier_accuracies.txt')
with tf.gfile.Open(accuracies_filename, 'w') as f:
    for tr, te in zip(train_accuracies, test_accuracies):
        # print('{0},{1}\n'.format(tr, te))
        f.write('{0},{1}\n'.format(tr, te))

fig, ax = plt.subplots()
plt.plot(train_accuracies, label='Train', alpha=0.5)
plt.plot(test_accuracies, label='Test', alpha=0.5)
plt.title("Accuracy")
plt.legend()
plt.savefig(os.path.join(statistics_dir, 'classifier_accuracies'))
plt.show()

