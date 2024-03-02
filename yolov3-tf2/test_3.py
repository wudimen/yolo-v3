'''
    使用tf.GradientTape
'''
import tensorflow as tf
from tqdm import tqdm

from yolov3_tf2.test.model_util import (
    YoloV3,
    YoloLoss
)

# from yolov3_tf2.models import (
#     YoloLoss
# )

from yolov3_tf2.test.util import (
    get_tfrecord_iterator
)
import numpy as np

learning_rate = 0.001
classes = 80
epochs = 10
batch_size = 1
batch_nums = 50
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# tf.compat.v1.config.experimental_run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

IMG_SIZE = 128
IMG_CHANNELS = 3

# tf.compat.v1.disable_eager_execution()
iterator = get_tfrecord_iterator("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
model = YoloV3(size=128, train=True)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# model.compile(optimizer=opt, loss=loss)


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # sess.run(iterator.initializer)
    one_batch = iterator.get_next()
    all_loss = 0
    pos = 0
    for epoch in range(epochs):
        for batch_pos in tqdm(range(batch_nums)):
            train_x, train_y = sess.run(one_batch)
            with tf.GradientTape() as tape:
                pred = model(train_x)
                loss_fns = [YoloLoss(anchors=yolo_anchors[mask], classes=classes) for mask in yolo_anchor_masks]
                regularization_loss = tf.reduce_sum(model.losses)
                loss = []
                loss.append(loss_fns[0](train_y[0], pred[0]))
                loss.append(loss_fns[1](train_y[1], pred[1]))
                loss.append(loss_fns[2](train_y[2], pred[2]))
                fina_loss = tf.reduce_sum(loss) + regularization_loss
                all_loss += fina_loss
                pos+=1
            grad = tape.gradient(fina_loss, model.trainable_variables)
            opt.apply_gradients(zip(grad, model.trainable_variables))
            print("loss=", sess.run(fina_loss))