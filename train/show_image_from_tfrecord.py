import cv2
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset

sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 1
SHUFFLE_SIZE = 1

train_tfrecord = "../data/images_test.tfrecords"
anchors = utils.get_anchors('../data/object_anchors.txt', IMAGE_H, IMAGE_W)
classes = utils.read_coco_names('../data/object.names')
# print(classes)
num_classes = len(classes)  # 识别的种类

parser = Parser(IMAGE_H, IMAGE_W, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)

is_training = tf.placeholder(tf.bool)
example = trainset.get_next()

for l in range(1):
    image, boxes = sess.run(example)
    # print(image)
    # print(sess.run(example))
    image, boxes = image[0], boxes[0]

    n_box = len(boxes)
    for i in range(n_box):
        image = cv2.rectangle(image, (int(float(boxes[i][0])),
                                      int(float(boxes[i][1]))),
                              (int(float(boxes[i][2])),
                               int(float(boxes[i][3]))), (255, 0, 0), 1)
        label = classes[boxes[i][4]]
        image = cv2.putText(image, label, (int(float(boxes[i][0])), int(float(boxes[i][1]))),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

    image = Image.fromarray(np.uint8(image))
    image.show()
