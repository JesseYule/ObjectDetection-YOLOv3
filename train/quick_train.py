# import sys
# sys.path.append("/data/videoAnalysis/simyolov3")

import os
import tensorflow as tf
from core import utils, YOLOv3
from core.dataset import dataset, Parser

sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 8
STEPS = 1
LR = 0.0001  # if Nan, set 0.0005, 0.0001
DECAY_STEPS = 100
DECAY_RATE = 0.9
SHUFFLE_SIZE = 45  # 这个要根据输入的训练图片数量进行修改！！！
CLASSES = utils.read_coco_names('../data/object.names')
ANCHORS = utils.get_anchors('../data/object_anchors.txt', IMAGE_H, IMAGE_W)
NUM_CLASSES = len(CLASSES)
EVAL_INTERNAL = 1
SAVE_INTERNAL = 1

# 在一个计算图开始前，将文件读入到queue中，TFRecord可以格式统一管理存储数据
train_tfrecord = "../data/images_train.tfrecords"
test_tfrecord = "../data/images_test.tfrecords"

parser = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_tfrecord, batch_size=BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset = dataset(parser, test_tfrecord, batch_size=BATCH_SIZE, shuffle=None)  # 这里我用全部测试集来训练，方便看效果


is_training = tf.placeholder(tf.bool)  # 占位符，运行时必须传入值

# 根据是is_training判断是不是在训练，然后调用trainset或者testset的get_next
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())


# y_true = [feature_map_1 , feature_map_2 , feature_map_3]
images, *y_true = example  # a,*c = 1,2,3,4   a=1, c = [2,3,4]
model = YOLOv3.yolov3(NUM_CLASSES, ANCHORS)


with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(images, is_training=is_training)
    loss = model.compute_loss(pred_feature_map, y_true)  # 计算loss值
    y_pred = model.predict(pred_feature_map)


# 显示标量信息
tf.summary.scalar("loss/coord_loss", loss[1])
tf.summary.scalar("loss/sizes_loss", loss[2])
tf.summary.scalar("loss/confs_loss", loss[3])
tf.summary.scalar("loss/class_loss", loss[4])

global_step = tf.Variable(0, trainable=False,
                          collections=[tf.GraphKeys.LOCAL_VARIABLES])  # 把变量添加到集合tf.GraphKeys.LOCAL_VARIABLES中

# merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可以显示训练时的各种信息了
write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("../log/train")
writer_test = tf.summary.FileWriter("../log/test")


# 恢复darknet-53特征提取器的权重参数, 只更新yolo-v3目标预测部分参数.
'''
saver_to_restore = tf.train.Saver(
    # 得到网络中所有可加载参数
    var_list=tf.contrib.framework.get_variables_to_restore(include=["yolov3/darknet-53"]))  # 固定特征提取器
'''

# 这里因为没办法加载DARKNET的参数，只能重新训练
update_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3", "yolov3/darknet-53"])

# 每一百次降低一次学习率, 学习率衰减
learning_rate = tf.train.exponential_decay(LR, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE,
                                           staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

# set dependencies for BN ops 设置BN操作的依赖关系
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # 从集合中取出变量

with tf.control_dependencies(update_ops):  # 在更新网络参数是,进行BN方差.等参数的更新
    train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

# 这里只是初始化参数，还没正式训练，一定要记住，对tensorflow，所有的代码都是定义，只有run，tensor才开始流动，graph才开始运行
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


saver = tf.train.Saver()

# 加载模型，继续训练
# saver.restore(sess, "../checkpoint/cpk-10000")

saver = tf.train.Saver(max_to_keep=2)  # 保存模型，最多保存两个

for step in range(STEPS):
    run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training: True})

    if True:
        # run_items[2] : boxes [Batch_size,10647,4], confs , probs
        # run_items[3] : feature_map_1 , feature_map_2 , feature_map_3
        train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])  # 放回查全率, 精确率

    # 写入日志
    writer_train.add_summary(run_items[1], global_step=step)
    writer_train.flush()  # Flushes the event file to disk 将事件文件刷新到磁盘

    # 保存模型
    if (step + 1) % SAVE_INTERNAL == 0:
        saver.save(sess, save_path="../checkpoint/cpk", global_step=step + 1)

    print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
          % (step + 1, run_items[5], run_items[6], run_items[7], run_items[8]))

    run_items = sess.run([write_op, y_pred, y_true] + loss, feed_dict={is_training: False})

    if (step + 1) % EVAL_INTERNAL == 0:

        test_rec_value, test_prec_value= utils.evaluate(run_items[1], run_items[2])

        print("\n=======================> evaluation result <================================\n")
        print("=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" % (step + 1, train_rec_value, train_prec_value))
        print("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" % (step + 1, test_rec_value, test_prec_value))
        print("\n=======================> evaluation result <================================\n")

    writer_test.add_summary(run_items[0], global_step=step)
    writer_test.flush()  # Flushes the event file to disk 写入磁盘

