# There are two ways of solving this problem.
# One is to have the matmul at the last layer output all 11 classes.
# The other is to output just 10 classes, and use a constant value of 0 for
# the logit for the last class. This still works because the softmax only needs
# n independent logits to specify a probability distribution over n + 1 categories.
# We implemented both solutions here.
extra_class = 1

is_NCHW = False
if is_NCHW:
    real_size = (3, 96, 128)
else:
    real_size = (96, 128, 3)
z_size = 100
learning_rate = 0.0003

data_dir = 'data'

tfrecord_filename = 'TFRecord_output'
train_num_shards = 10
test_num_shards = 4

buffer_size = 1024
batch_size = 8
epochs = 2

statistics_dir = 'statistics'
every_steps_to_save = 2
ckpt_dir = 'checkpoints'
generated_dir = 'generated_img'