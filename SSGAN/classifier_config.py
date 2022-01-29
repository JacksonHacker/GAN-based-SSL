is_NCHW = False

if is_NCHW:
    input_size = (3, 96, 128)
else:
    input_size = (96, 128, 3)

learning_rate = 0.0003

data_dir = 'data_for_classifier'

tfrecord_filename = 'TFRecord_output'
train_num_shards = 10
test_num_shards = 4

buffer_size = 1024
batch_size = 64
epochs = 10

statistics_dir = 'statistics_for_classifier'
every_steps_to_save = 10
ckpt_dir = 'checkpoints_for_classifier'