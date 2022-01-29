import os
import matplotlib.pyplot as plt
import tensorflow as tf
from model import GAN
from train import train
from os.path import isfile, isdir



def scale(x, feature_range=(-1, 1)):

    x = tf.cast(x, tf.float32)
    x = tf.truediv(x, 255.)
    x = tf.multiply(x, 2.)
    x = x - 1.
    return x

def parse(serialized):
    features = \
        {
            'image/encoded':   tf.FixedLenFeature([], tf.string),
            'image/label':     tf.FixedLenFeature([], tf.int64),
            'image/label_mask':tf.FixedLenFeature([], tf.int64)
        }

    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    image_raw = parsed_example['image/encoded']
    image_unit8 = tf.image.decode_jpeg(image_raw, channels=3)
    label = parsed_example['image/label']
    label_mask = parsed_example['image/label_mask']


    return scale(image_unit8), label, label_mask

def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

data_dir = 'data'

if not isdir(data_dir):
    raise Exception("Data directory doesn't exist!")


real_size = (96,128,3)
z_size = 100
learning_rate = 0.0003

tfrecord_filename = 'TFRecord_output'
num_shards = 4
train_filename_list = [_get_dataset_filename(data_dir, "train", i, tfrecord_filename, num_shards) for i in range(num_shards)]
test_filename_list = [_get_dataset_filename(data_dir, "test", i, tfrecord_filename, num_shards) for i in range(num_shards)]
print(train_filename_list)
print(test_filename_list)

for file in train_filename_list:
    if not isfile(file):
        raise Exception("TFRecord file {0} doesn't exist!".format(file))

for file in test_filename_list:
    if not isfile(file):
        raise Exception("TFRecord file {0} doesn't exist!".format(file))

dataset_train = tf.data.TFRecordDataset(train_filename_list)
dataset_test = tf.data.TFRecordDataset(test_filename_list)
dataset_train = dataset_train.map(parse)
dataset_test = dataset_test.map(parse)

buffer_size = 1024
batch_size = 16
epochs = 10

dataset_train.shuffle(buffer_size=buffer_size)
dataset_train = dataset_train.batch(batch_size)
dataset_test = dataset_test.batch(batch_size)

iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                           dataset_train.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(dataset_train)
test_init_op = iterator.make_initializer(dataset_test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        print("Epoch", e)

        num_examples = 0
        num_correct = 0

        sess.run(training_init_op)
        while True:
            try:
                x, y, label_mask = sess.run(next_element)
                # print('y: {0};  label_mask: {1}'.format(y, label_mask))
                # print('#####################################')
            except tf.errors.OutOfRangeError:
                break