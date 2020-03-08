import tensorflow as tf

'''
华为不支持的算子都在这里重新写一遍
'''


def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * tf.add(1., leak)
        f2 = 0.5 * (1. - leak)
        return tf.add(f1 * x, f2 * tf.abs(x))


def bn_transform(x):
    # 因为华为对算子的支持十分不完善，这里暂时写一个简单的batch normalization，还需要改进
    # 计算整个tensor的mean和var，然后对整个tensor做标准化处理

    size = x.shape.as_list()

    tensor_num = size[1]*size[2]*size[3]

    batch_mean = tf.reduce_sum(x) / tensor_num  # 计算tensor的mean
    batch_var = tf.reduce_sum(tf.square(tf.subtract(x, batch_mean))) / tensor_num  # 计算tensor的var

    result = (x - batch_mean) / tf.sqrt(batch_var)

    return result

