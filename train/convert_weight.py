import os
import sys
import time
import argparse
import tensorflow as tf
from core import yolov3, utils


class parser(argparse.ArgumentParser):

    def __init__(self, description):
        super(parser, self).__init__(description)

        # 这里要根据读取的文件名修改
        self.add_argument(
            "--ckpt_file", "-cf", default='../checkpoint/cpk-1', type=str,
            help="[default: %(default)s] The checkpoint file ...",
            metavar="<CF>",
        )

        # 这里的num_calsses要根据实际情况修改，记得修改！！！
        self.add_argument(
            "--num_classes", "-nc", default=1, type=int,
            help="[default: %(default)s] The number of classes ...",
            metavar="<NC>",
        )

        self.add_argument(
            "--anchors_path", "-ap", default="../data/coco_anchors.txt", type=str,
            help="[default: %(default)s] The path of anchors ...",
            metavar="<AP>",
        )

        self.add_argument(
            "--weights_path", "-wp", default='../checkpoint/yolov3.weights', type=str,
            help="[default: %(default)s] Download binary file with desired weights",
            metavar="<WP>",
        )

        self.add_argument(
            "--convert", "-cv", action='store_false',
            help="[default: %(default)s] Downloading yolov3 weights and convert them",
        )

        self.add_argument(
            "--freeze", "-fz", action='store_false',
            help="[default: %(default)s] freeze the yolov3 graph to pb ...",
        )

        self.add_argument(
            "--image_h", "-ih", default=416, type=int,
            help="[default: %(default)s] The height of image, 416 or 608",
            metavar="<IH>",
        )

        self.add_argument(
            "--image_w", "-iw", default=416, type=int,
            help="[default: %(default)s] The width of image, 416 or 608",
            metavar="<IW>",
        )

        self.add_argument(
            "--iou_threshold", "-it", default=0.5, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<IT>",
        )

        self.add_argument(
            "--score_threshold", "-st", default=0.5, type=float,  # 分数阈值
            help="[default: %(default)s] The score_threshold for gpu nms",
            metavar="<ST>",
        )


def main(argv):
    flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
    print("=> the input image size is [%d, %d]" % (flags.image_h, flags.image_w))
    anchors = utils.get_anchors(flags.anchors_path, flags.image_h, flags.image_w)
    # print(anchors)
    # exit()
    model = yolov3.yolov3(flags.num_classes, anchors)

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        inputs = tf.placeholder(tf.float32, [1, flags.image_h, flags.image_w, 3])  # placeholder for detector inputs
        print("=>", inputs)

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs, is_training=False)  # 返回3个尺度的feature_map

        # 获取网络给出绝对boxes(左上角,右下角)信息, 未经过最大抑制去除多余boxes
        boxes, confs, probs = model.predict(feature_map)
        scores = confs * probs
        print("=>", boxes.name[:-2], scores.name[:-2])
        # cpu 运行是恢复模型所需要的网络节点的名字
        cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
        boxes, scores, labels = utils.gpu_nms(boxes, scores, flags.num_classes,
                                              score_thresh=flags.score_threshold,
                                              iou_thresh=flags.iou_threshold)
        print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
        # gpu 运行是恢复模型所需要的网络节点的名字 , 直接运算得出最终结果
        gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))
        our_out_node_names = ["yolov3/yolo-v3/feature_map_1", "yolov3/yolo-v3/feature_map_2", "yolov3/yolo-v3/feature_map_3"]

        # 只有第一次需要读取预训练参数yolov3.weights

        # if flags.convert:
        #     load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), flags.weights_path)
        #     sess.run(load_ops)
        #     save_path = saver.save(sess, save_path=flags.ckpt_file)
        #     print('=> model saved in path: {}'.format(save_path))

        # print(flags.freeze)
        if flags.freeze:
            saver.restore(sess, flags.ckpt_file)
            print('=> checkpoint file restored from ', flags.ckpt_file)
            utils.freeze_graph(sess, '../checkpoint/yolov3_cpu_nms.pb', cpu_out_node_names)
            utils.freeze_graph(sess, '../checkpoint/yolov3_gpu_nms.pb', gpu_out_node_names)
            utils.freeze_graph(sess, '../checkpoint/ouryolov3.pb', our_out_node_names)


if __name__ == "__main__": main(sys.argv)
