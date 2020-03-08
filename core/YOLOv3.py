import tensorflow as tf
from core import CONV
from core import DARKNET
from core.unsupportedOperator import *

slim = tf.contrib.slim


class yolov3(object):

    def __init__(self, num_classes, anchors,
                 batch_norm_decay=0.9):
        '''
        :param num_classes: class
        :param anchors: number of anchors 列表
        :param batch_norm_decay:
        '''
        # self._ANCHORS =
        #               [[10 ,13], [16 , 30], [33 , 23],
        #               [30 ,61], [62 , 45], [59 ,119],
        #               [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._NUM_CLASSES = num_classes
        self.feature_maps = []  # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    def _yolo_block(self, inputs, filters, is_training=False):
        # if stride > 1 , padding
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 1, 1, is_training=is_training)
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 2, 3, is_training=is_training)
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 1, 1, is_training=is_training)
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 2, 3, is_training=is_training)
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 1, 1, is_training=is_training)
        route = inputs
        inputs = CONV._conv2d_fixed_padding(inputs, filters * 2, 3, is_training=is_training)

        # 这里也是返回两个尺寸的feature map
        return route, inputs

    # 目标识别的层, 转换到合适的深度（也就是通道）,以满足不同class_num数据的分类
    # 模型输出层，也就是最后一层，输出所有anchors预测的box的坐标长宽、各个class的概率
    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)

        # 注意那个num_outputs是指卷积核的数量，也就是通道的数量，用卷积层替代全脸层的奥义就在于用一个通道表示一个结果
        feature_map = slim.conv2d(inputs, num_outputs=num_anchors * (5 + self._NUM_CLASSES), kernel_size=1,
                                  stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        return feature_map

    # 将网络计算的的缩放量和偏移量与anchors、网格位置结合,得到在原图中的绝对位置与大小
    def _reorg_layer(self, feature_map, anchors):

        # 将张量转换为适合的格式
        num_anchors = len(anchors)  # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]  # 网格数
        # the downscale image in height and weight

        # 这里的img_size和grid_size都是二维的（w，h），除得的结果也是二维的
        stride = tf.cast(tf.round(tf.divide(tf.cast(self.img_size, tf.float32), tf.cast(grid_size, tf.float32))),
                         tf.float32)


        # feature的输入是（batch，cell，cell，通道），一共思维，通道数等于anchor*（5+class），
        # 这里把通道的维进行拆分， 变成（anchor，（5+class））
        # 但是因为reshape算子华为不支持5维输入，所以这里的操作稍微麻烦一点先降维再升维

        feature_map = tf.reshape(feature_map,
                                 [-1, grid_size[0], grid_size[1], num_anchors * (5 + self._NUM_CLASSES)])

        feature_map31 = tf.expand_dims(feature_map[:, :, :, 0:(5 + self._NUM_CLASSES)], axis=3)
        feature_map32 = tf.expand_dims(feature_map[:, :, :, (5 + self._NUM_CLASSES):2*(5 + self._NUM_CLASSES)], axis=3)
        feature_map33 = tf.expand_dims(feature_map[:, :, :, 2*(5 + self._NUM_CLASSES):3*(5 + self._NUM_CLASSES)], axis=3)

        feature_map = tf.concat([feature_map31, feature_map32, feature_map33], axis=3)

        # box_centers, box_sizes, conf_logits, prob_logits = tf.split(
        #     feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)
        # 对最后一个维度进行切割，最后一个维度尺寸为5+class，切成2+2+1+class，第一个2是box坐标，第二个2是box长宽，第三个1是置信度（IOU）

        box_centers = feature_map[:, :, :, :, 0:2]
        box_sizes = feature_map[:, :, :, :, 2:4]
        conf_logits = feature_map[:, :, :, :, 4:5]
        prob_logits = feature_map[:, :, :, :, 5:]

        box_centers = tf.nn.sigmoid(box_centers)  # 使得偏移量变为非负,且在0~1之间, 超过1之后,中心点就偏移到了其他的单元中

        # 写出各个grid的左上角坐标
        grid_x = tf.range(grid_size[1], dtype=tf.float32)
        grid_y = tf.range(grid_size[0], dtype=tf.float32)

        a, b = tf.meshgrid(grid_x, grid_y)  # 构建网格 https://blog.csdn.net/MOU_IT/article/details/82083984
        '''
        a=[0,5,10]
        b=[[[0],[5]],[15,20,25]]
        A,B=tf.meshgrid(a,b)
        with tf.Session() as sess:
          print (A.eval())
          print (B.eval())

        结果：
        [[ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]
         [ 0  5 10]]
        [[ 0  0  0]
         [ 5  5  5]
         [15 15 15]
         [20 20 20]
         [25 25 25]]
         '''
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # 组合生产每个单元格左上角的坐标, 排列组合
        '''
        [0,0]
        [0,1]
        [0,2]
        .....
        [1,0]
        [1,1]
        .....
        [12,12]
        '''

        x_y_offset = tf.cast(x_y_offset, tf.float32)
        # 恢复成5x5x1x2 的张量，也就是每个grid(一共5*5)一个坐标
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])


        # 华为不支持add算子5维广播
        # 先降维，再相加，再拼接升维

        # 因为有三个anchor，所以加三次，这里直接拼接成新的维度，使维度对应起来
        x_y_offset = tf.concat([x_y_offset, x_y_offset, x_y_offset], axis=2)

        # 本来应该是（batch，cell，cell，num_anchors， 2），最后两个维度对应相加，但是因为box_center有5维，x_y_offset有4维（少了batch）
        # 华为不支持这样的广播加法
        # 所以只能把维度降下来，变成（batch，cell，cell，num_anchors*2）和（cell，cell，num_anchors*2）进行广播加法
        # 最后再升维，变回（batch，cell，cell，num_anchors，2）
        box_centers = tf.reshape(box_centers, [-1, grid_size[0], grid_size[1], num_anchors * 2])
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], num_anchors * 2])

        box_centers = tf.add(box_centers, x_y_offset)  # 物体的中心坐标

        box_centers31 = tf.expand_dims(box_centers[:, :, :, 0:2], axis=3)
        box_centers32 = tf.expand_dims(box_centers[:, :, :, 2:4], axis=3)
        box_centers33 = tf.expand_dims(box_centers[:, :, :, 4:6], axis=3)

        box_centers = tf.concat([box_centers31, box_centers32, box_centers33], axis=3)

        # 把x_y_offset的维度也变回去，不然最后输出就和一开始预设的不一样了
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], num_anchors, 2])

        # 接下来对结果进行缩放，也就是对上面计得的坐标值乘一个系数stride_conc（因为之前我们假设了每个cell的宽高都是1）
        # 同样因为华为不支持5维广播乘法，这里也要先降维计算后再升维
        box_centers = tf.reshape(box_centers, [-1, grid_size[0], grid_size[1], num_anchors * 2])
        stride_conc = tf.concat([stride[::-1],  stride[::-1], stride[::-1]], axis=0)

        box_centers = tf.multiply(box_centers, stride_conc)

        box_centers31 = tf.expand_dims(box_centers[:, :, :, 0:2], axis=3)
        box_centers32 = tf.expand_dims(box_centers[:, :, :, 2:4], axis=3)
        box_centers33 = tf.expand_dims(box_centers[:, :, :, 4:6], axis=3)

        box_centers = tf.concat([box_centers31, box_centers32, box_centers33], axis=3)

        # 在原图的坐标位置,反归一化 [h,w] -> [y,x]，因为一开始grid的坐标是整数，比如【0，1】、【1，1】，现在算出了中心坐标，
        # 比如[1.3,3.4]，现在就要把这个坐标放大到原图的尺寸


        # box_sizes就是（w，h），是一开始feature计算的，还没有处理过，现在就用exp函数处理一下，详情看论文
        # tf.exp(box_sizes) 避免缩放出现负数, box_size[13,13,3,2], anchor[3,2]
        box_sizes = tf.exp(box_sizes)   # anchors -> [w, h] 使用网络计算出的缩放量对anchors进行缩放

        # 同样，因为华为不支持广播乘法，所以也是先降维计算后再升维
        box_sizes = tf.reshape(box_sizes, [-1, grid_size[0], grid_size[1], num_anchors * 2])

        # anchor本身有一个预设的（w，h），现在就是要和模型计算的box_size的（w，h）做乘法（详情看论文）
        anchors = tf.reshape(anchors, [6])

        box_sizes = tf.multiply(box_sizes, tf.cast(anchors, tf.float32))

        box_sizes31 = tf.expand_dims(box_sizes[:, :, :, 0:2], axis=3)
        box_sizes32 = tf.expand_dims(box_sizes[:, :, :, 2:4], axis=3)
        box_sizes33 = tf.expand_dims(box_sizes[:, :, :, 4:6], axis=3)

        box_sizes = tf.concat([box_sizes31, box_sizes32, box_sizes33], axis=3)

        boxes = tf.concat([box_centers, box_sizes], axis=-1)  # 拼接坐标以及宽高

        return x_y_offset, boxes, conf_logits, prob_logits

    @staticmethod  # 静态静态方法不睡和类和实例进行绑定
    def _upsample(inputs, out_shape):  # 上采样, 放大图片
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))  # 使用最近邻改变图像大小
        inputs = tf.identity(inputs, name='upsampled')

        return inputs

    # 前向传播,得到3个feature_map
    def forward(self, inputs, is_training=False, reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
               Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on 他在稍后将被需要
        self.img_size = tf.shape(inputs)[1:3]

        # set batch norm params
        batch_norm_params = {
            # 'decay': self._BATCH_NORM_DECAY,  # https://www.cnblogs.com/hellcat/p/8058092.html
            'epsilon': 1e-05,
            'scale': 1.,
            'offset': 1e-6,
            'is_training': is_training,
            # 'fused': None,  # Use fused batch norm if possible.
        }

        # arg_scope为slim.conv2d等函数的参数提供默认值，简化了代码
        with slim.arg_scope([slim.conv2d, CONV._fixed_padding], reuse=reuse):

            with slim.arg_scope([slim.conv2d],
                                # normalizer_fn=tf.nn.fused_batch_norm,
                                # normalizer_params=batch_norm_params,
                                normalizer_fn=lambda x: bn_transform(x),
                                biases_initializer=None,
                                activation_fn=lambda x: LeakyRelu(x)):

                with tf.variable_scope('darknet-53'):

                    route_1, route_2, inputs = DARKNET.darknet53(inputs).outputs
                    # darknet会返回三个尺度的feature map，然后YOLO3其实也是利用这三个feature map，分别计算一次，
                    # 最后又返回三个feature map
                    # route_1 : 52x52x256
                    # route_2 : 26x26x512
                    # inputs  : 13x13x1024

                with tf.variable_scope('yolo-v3-FRN'):

                    # 以下是FRN架构，可对着FRN架构的图看下面的代码

                    # 首先对input输入到yolo block，得到第一个输出feature_map_1
                    route, inputs = self._yolo_block(inputs, filters=512, is_training=is_training)
                    feature_map_1 = self._detection_layer(inputs, anchors=self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    # input输入到yolo后，再卷积和上采样，为下一个输出feature_map_2做准备
                    inputs = CONV._conv2d_fixed_padding(route, 256, 1, is_training=is_training)
                    upsample_size = route_2.get_shape().as_list()
                    #  52x52 --> 26x26
                    inputs = self._upsample(inputs, upsample_size)  # 通过直接放大进行上采样
                    inputs = tf.concat([inputs, route_2], axis=3)
                    route, inputs = self._yolo_block(inputs, filters=256, is_training=is_training)
                    feature_map_2 = self._detection_layer(inputs, anchors=self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    # 这里也是和上面差不多的操作，也是卷积、上采样、拼接、yolo处理、得到结果
                    inputs = CONV._conv2d_fixed_padding(route, 128, 1, is_training=is_training)
                    upsample_size = route_1.get_shape().as_list()
                    # 26x26 --> 52x52
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)
                    route, inputs = self._yolo_block(inputs, filters=128, is_training=is_training)
                    feature_map_3 = self._detection_layer(inputs, anchors=self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            # 返回多个feature map，多个尺度的信息
            return feature_map_1, feature_map_2, feature_map_3

    def _reshape(self, x_y_offset, boxes, confs, probs):
        # 构成一个(batch_size, cell*cell*len(anchors) , boxes)
        grid_size = x_y_offset.shape.as_list()[:2]  # 网格数
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])  # 3个anchor
        confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])  # 3个anchor分别对应概率
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self._NUM_CLASSES])  # 类别概率

        return boxes, confs, probs

    # 上面只是计算了feature map，加下来就是从feature map中提取我们需要的信息（边界框信息、置信度、分类结果）
    def predict(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              由给出的feature map 计算
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        # 预设使用三个anchors，然后一共有三种不同的尺寸的feature map，每个feature map又对应三种anchors，所有anchors一共有九个
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3])]

        # reorg_layer返回x_y_offset, boxes, conf_logits, prob_logits
        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        boxes_list, confs_list, probs_list = [], [], []

        for result in results:

            # *result =  x_y_offset, boxes, confs, probs
            # 这里的_reshape函数是上面定义的，直接提取出boxes, confs, probs
            boxes, conf_logits, prob_logits = self._reshape(*result)
            # --> (batch_size, cell*cell*anchor_num, boxes/conf/prob)

            confs = tf.sigmoid(conf_logits)  # 转化成概率
            probs = tf.sigmoid(prob_logits)  # 转化成概率,每种类和不在为0

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # 将3个feature_map中所有的信息,整合到一个张量
        # shape : [Batch_size,10647,4]  10647 = 13x13x3 + 26x26x3 + 52x52x3
        boxes = tf.concat(boxes_list, axis=1)  # [Batch_size,10647,4]
        confs = tf.concat(confs_list, axis=1)  # [Batch_size,10647,1]
        probs = tf.concat(probs_list, axis=1)  # [Batch_size,10647,class_num]

        # 坐标转化:中心坐标转化为 左上角坐标,右下角坐标 --> 方便计算矩形框

        center_x = boxes[:, :, 0:1]

        center_y = boxes[:, :, 1:2]
        width = boxes[:, :, 2:3]
        height = boxes[:, :, 3:4]

        x0 = center_x - tf.divide(width, 2.)
        y0 = center_y - tf.divide(height, 2.)

        x1 = tf.add(center_x, tf.divide(width, 2.))
        y1 = tf.add(center_y, tf.divide(height, 2.))

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def compute_loss(self, pred_feature_map, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """
        :param pred_feature_map: list [feature_map_1,feature_map_2,feature_map3]
                feature_map_1[13,13,3,(5 + self._NUM_CLASSES)]
        :param y_true: list [y_true_13, y_true_26, y_true_52]
               y_true_13 [13,13,3,(5 + self._NUM_CLASSES)] 只有含有目标的网格中存在信息,其余均为0.
        :param ignore_thresh: 0.5
        :param max_box_per_image:
        :return:
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.
        # total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        # 计算每个featurn_map的损失
        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_true[i], _ANCHORS[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def loss_layer(self, feature_map_i, y_true, anchors):

        # y_ture [13,13,3,5+class_id] 13*13 grid，3 anchors
        # size in [h, w] format! don't get messed up!

        grid_size = tf.shape(feature_map_i)[1:3]  # 提取cell*cell的size
        grid_size_ = feature_map_i.shape.as_list()[1:3]  # 提取cell*cell的size

        # 本身具有[-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES]的shape,
        # 但在进过tf.py_func方法时丢失shape信息,使用reshape重新赋予shape
        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES])

        # the downscale ratio in height and weight
        ratio = tf.cast(tf.divide(tf.cast(self.img_size, tf.float32), tf.cast(grid_size, tf.float32)), tf.float32)

        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        # 进过self._reorg_layer后会boxe会被换成绝对位置, 会使用ratio进行换算到cellxcell上
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]

        object_mask = y_true[..., 4:5]  # 该feature_map下所有的目标,有目标的为1,无目标的为0

        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box, 该feature_map下所有检测目标的数量

        # boolean_mask(a,b) 将使a矩阵仅保留与b中“True”元素同下标的部分
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4],
                                           tf.cast(object_mask[..., 0], 'bool'))  # 获取有每个(3个)anchor的中心坐标,长宽

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou 计算每个pre_boxe与所有true_boxe的交并比.
        # true:[V,2],[V,2]
        # pre : [13,13,3,2]
        # out_shape: [N, 13, 13, 3, V],
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # iou_shape : [N,13,13,3,V] 每个单元下每个anchor与所有的true_boxes的交并比
        best_iou = tf.reduce_max(iou, axis=-1)  # 选择每个anchor中iou最大的那个.
        # out_shape : [N,13,13,3]

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)  # 如果iou低于0.5将会丢弃此anchor\
        # out_shape : [N,13,13,3] 0,1张量

        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # out_shape: [N, 13, 13, 3, 1] 0,1张量

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]  # 坐标反归一化

        # 之前对模型输出的坐标宽高做了变换，这里对真实数据和模型输出一起做逆变换，为的是计算模型原始输出的loss(有正负的)
        true_xy = tf.cast(tf.divide(tf.cast(y_true[..., 0:2], tf.float32), tf.cast(ratio[::-1], tf.float32)) - x_y_offset, tf.float32)  # 绝对(image_size * image_size)信息 转换为 单元(cellxcell)相对信息
        pred_xy = tf.cast(tf.divide(tf.cast(pred_box_xy, tf.float32), tf.cast(ratio[::-1], tf.float32)) - x_y_offset, tf.float32)  # 获取网络真实输出值

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2],
        true_tw_th = tf.cast(tf.divide(tf.cast(y_true[..., 2:4], tf.float32), tf.cast(anchors, tf.float32)), tf.float32)
        pred_tw_th = tf.cast(tf.divide(tf.cast(pred_box_wh, tf.float32), tf.cast(anchors, tf.float32)), tf.float32)
        # for numerical stability 稳定训练, 为0时不对anchors进行缩放, 在模型输出值特别小是e^out_put为0
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        # 还原网络最原始的输出值(有正负的)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # 较小的面接的box有较大的权重
        # shape: [N, 13, 13, 3, 1]  2. - 面积   为1时表示保持原始权重
        box_loss_scale = tf.cast(2. - (tf.divide(tf.cast(y_true[..., 2:3], tf.float32), tf.cast(self.img_size[1], tf.float32))) * (
                tf.divide(tf.cast(y_true[..., 3:4], tf.float32), tf.cast(self.img_size[0], tf.float32))), tf.float32)

        # shape: [N, 13, 13, 3, 1] 方框损失值, 中心坐标均方差损失 * mask[N, 13, 13, 3, 1]
        # 仅仅计算有目标单元的loss, 不计算那些错误预测的boxes, 在预测是首先会排除那些conf,iou底的单元
        xy_loss = tf.cast(tf.divide(tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale), N), tf.float32)  # N:batch_size
        wh_loss = tf.cast(tf.divide(tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale), N), tf.float32)

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask  # 只要存在目标的box
        conf_neg_mask = (1 - object_mask) * ignore_mask  # 选择不存在目标,同时iou小于阈值(0.5),

        # 分离正样本和负样本
        # 正样本损失
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # 处理后的负样本损失,只计算那些是单元格中没有目标,同时IOU小于0.5的单元,
        # 只惩罚IOU<0.5,而不惩罚IOU>0.5 的原因是可能该单元内是有目标的,仅仅只是目标中心点却没有落在该单元中.所以不计算该loss
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)

        conf_loss = tf.cast(tf.divide(tf.reduce_sum(tf.add(conf_loss_pos, conf_loss_neg)), N), tf.float32)  # 平均交叉熵,同时提高正确分类,压低错误分类

        # shape: [N, 13, 13, 3, 1], 分类loss
        # boject_mask 只看与anchors相匹配的anchors
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=pred_prob_logits)
        class_loss = tf.cast(tf.divide(tf.reduce_sum(class_loss), N), tf.float32)

        return xy_loss, wh_loss, conf_loss, class_loss

    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match 只关心大小的匹配
        '''
        # shape:
        # true_box_??: [V, 2] V:目标数量
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2] , 扩张维度方便进行维度广播
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2] V:该尺度下分feature_map 下所有的目标是目标数量
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] --> [N, 13, 13, 3, V, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2] 维度广播
        # 真boxe,左上角,右下角, 假boxe的左上角,右小角,
        intersect_mins = tf.cast(tf.maximum(pred_box_xy - tf.divide(pred_box_wh, 2.),
                                    true_box_xy - tf.divide(true_box_wh, 2.)), tf.float32)

        intersect_maxs = tf.cast(tf.minimum(tf.divide(tf.add(pred_box_xy, pred_box_wh), 2.),
                                    tf.divide(tf.add(true_box_xy, true_box_wh), 2.)), tf.float32)
        # tf.maximun 去除那些没有面积交叉的矩形框, 置0
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)  # 得到重合区域的长和宽

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 重合部分面积
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # 预测区域面积
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # 真实区域面积
        # [N, 13, 13, 3, V]
        iou = tf.cast(tf.divide(intersect_area, (pred_box_area + true_box_area - intersect_area)), tf.float32)

        return iou
