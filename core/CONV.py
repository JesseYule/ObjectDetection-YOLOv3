import tensorflow as tf
from core.unsupportedOperator import *

slim = tf.contrib.slim


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1, is_training=False):

    # 步长大于1就要padding
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)

    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))  # 普通的卷积运算

    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    演空间维度填充输入,与输入大小无关, 只有与所使用的卷积核有关,左右两边进行填充

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    # padding的目的是使得卷积核能覆盖原始的输入，覆盖所有的边界，而不是为了让输入padding后能刚好让卷积核进行计算
    # 所以padding的大小为卷积核大小减1，这是让边界能被任意卷积核、在任意步长都能覆盖的最小padding尺寸
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # anchor那个维度和通道不需要padding，只需要对两个维度padding就可以
    # 假设[pad_beg, pad_end]=[1,2]，表示向维度的一边填充1位，另一边填充2位
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)

    return padded_inputs
