import tensorflow as tf
from core import CONV

slim = tf.contrib.slim


class darknet53(object):

    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    def _darknet53_block(self, inputs, filters):
        '''
            Darknet网络主要由一些列1*1和3*3的卷积层组成，
            所以这里就写了一个block，把1*1和3*3的卷积层构成一个模块
        '''
        shortcut = inputs
        inputs = CONV._conv2d_fixed_padding(inputs, filters=filters * 1, kernel_size=1)
        inputs = CONV._conv2d_fixed_padding(inputs, filters=filters * 2, kernel_size=3)

        inputs = tf.add(inputs, shortcut)  # 把卷积运算结果和原输入拼接在一起，即残差连接

        return inputs

    def forward(self, inputs):

        inputs = CONV._conv2d_fixed_padding(inputs, 32, 3, strides=1)
        inputs = CONV._conv2d_fixed_padding(inputs, 64, 3, strides=2)  # 208
        inputs = self._darknet53_block(inputs, 32)  #
        inputs = CONV._conv2d_fixed_padding(inputs, 128, 3, strides=2)  # 104

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        inputs = CONV._conv2d_fixed_padding(inputs, 256, 3, strides=2)  # 52

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs  # 注意这里，因为模型会输出不同尺寸的输出，这里的输出尺寸就比较大
        inputs = CONV._conv2d_fixed_padding(inputs, 512, 3, strides=2)  # 26

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs  # 这里的输出比route_1的尺寸就更小了，但仍然比后面的input要大
        inputs = CONV._conv2d_fixed_padding(inputs, 1024, 3, strides=2)  # 13

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        # 注意最后darknet53返回了三个feature map，返回了多个尺度的信息进行进一步的分析
        return route_1, route_2, inputs
