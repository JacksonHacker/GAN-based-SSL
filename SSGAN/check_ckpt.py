from tensorflow.python.tools import inspect_checkpoint as ckpt

ckpt.print_tensors_in_checkpoint_file('./checkpoints/final.ckpt-6', tensor_name='', all_tensors=True)