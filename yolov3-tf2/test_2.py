'''
    使用tf.placeholder
    结果：tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution. 
            Use Eager execution or decorate this function with @tf.function.
'''
import tensorflow as tf

from yolov3_tf2.test.model_util import (
    YoloV3,
    YoloLoss
)

# from yolov3_tf2.models import (
#     YoloV3,
#     YoloLoss
# )
from yolov3_tf2.test.util import (
    get_tfrecord_iterator
)
import numpy as np

learning_rate = 0.001
classes = 80
epochs = 1
batch_size = 1
batch_nums = 50
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])



IMG_SIZE = 128
IMG_CHANNELS = 3

# tf.compat.v1.disable_eager_execution()
iterator = get_tfrecord_iterator("C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/voc2012_val.tfrecord")
model = YoloV3(size=128, train=True)

loss_fns = [YoloLoss(anchors=yolo_anchors[mask], classes=classes) for mask in yolo_anchor_masks]
input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
label0 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE // 32, IMG_SIZE // 32, len(yolo_anchor_masks[0]), 6])     # [batch_size, grid, grid, anchors, (6)]
label1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE // 16, IMG_SIZE // 16, len(yolo_anchor_masks[1]), 6])
label2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE // 8, IMG_SIZE // 8, len(yolo_anchor_masks[2]), 6])
[pred0, pred1, pred2] = model(input)            # [batch_size, grid, grid, anchors, classes+5] * 3

loss = []
loss.append(loss_fns[0](label0, pred0))
loss.append(loss_fns[1](label1, pred1))
loss.append(loss_fns[2](label2, pred2))
loss = tf.reduce_sum(loss)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

print(loss)

model.compile(optimizer=opt, loss=loss)

pred = model(input)


with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    one_batch = iterator.get_next()
    for epoch in range(epochs):
        for batch_num in range(batch_nums):
            train_x, train_y = sess.run(one_batch)
            sess.run(opt, feed_dict={input:train_x, label0:train_y[0], label1:train_y[1], label2:train_y[2]})
            print("loss=", loss.eval(feed_dict={input:train_x, label0:train_y[0], label1:train_y[1], label2:train_y[2]}))
