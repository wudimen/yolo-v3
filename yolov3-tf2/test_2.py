import tensorflow as tf



from yolov3_tf2.test.model_util import (
    YoloV3,
    YoloLoss
)
from yolov3_tf2.test.util import (
    get_tfrecord_iterator
)

IMG_SIZE = 128
IMG_CHANNELS = 3

iterator = get_tfrecord_iterator("C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/voc2012_val.tfrecord")
model = YoloV3(size=128)

input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
output = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ])

pred = model(input)

loss_fn = [YoloLoss(anchors=)]

with tf.compat.v1.Session() as sess:

