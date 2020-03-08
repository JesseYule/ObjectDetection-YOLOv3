import tensorflow as tf
from core import utils, YOLOv3
import cv2
from PIL import Image
import numpy as np

input_image = "../testpic/1.jpg"
image = Image.open(input_image)

image_resize = cv2.resize(np.array(image) / 255., (416, 416))
image_place = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
CLASSES = utils.read_coco_names('../data/object.names')
ANCHORE = utils.get_anchors("../data/object_anchors.txt", 416, 416)
model = YOLOv3.yolov3(len(CLASSES), ANCHORE)

with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(image_place, is_training=False)
    print(pred_feature_map)
    pred = model.predict(pred_feature_map)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "../checkpoint/cpk-10000")

boxes, confs, prods = sess.run(pred, feed_dict={image_place: np.expand_dims(image_resize, 0)})
boxes, confs, prods = utils.cpu_nms(boxes, confs * prods, len(CLASSES))
utils.draw_boxes(image, boxes, confs, prods, CLASSES, (416, 416), "../font/HuaWenXinWei-1.ttf")
print(boxes, confs, prods)