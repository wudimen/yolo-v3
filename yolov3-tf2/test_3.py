'''
    使用tf.GradientTape
'''
import tensorflow as tf
from tqdm import tqdm
import cv2

import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from yolov3_tf2.test.util import (
    load_darknet_weights,
    freeze_all,
    transform_image,
    draw_bbox_with_float_and_imgdata,
    draw_output_img,
)

from yolov3_tf2.test.model_util import (
    YoloV3,
    YoloLoss,
)

# from yolov3_tf2.models import (
#     YoloV3
# )

from yolov3_tf2.test.util import (
    get_tfrecord_iterator
)
import numpy as np


learning_rate = 0.001
epochs = 10
num_classes = 80
batch_size = 1
batch_nums = 50
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# tf.compat.v1.config.experimental_run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

IMG_SIZE = 128 * 4
IMG_CHANNELS = 3

# def load_model(weights, transfer='none'):
#     model = YoloV3(size=IMG_SIZE, train=True, classes=num_classes)
#     # anchors = yolo_anchors
#     # anchor_masks = yolo_anchor_masks

#     if transfer == 'none':
#         pass
#     elif transfer in ['darknet', 'no_output']:
#         model_pretrained = YoloV3(size=IMG_SIZE, trian=True, classes=num_classes)
#         model_pretrained.load_weights(weights)

#         if transfer == 'darknet':
#             model.get_layers('yolo_darknet').set_weights(model_pretrained.get_layers('yolo_darknet').get_weights())
#             freeze_all(model.get_layers('yolo_darknet'))
#         elif transfer == 'no_output':
#             for l in model.layers:
#                 if not l.name.startswitch('yolo_output'):
#                     l.set_weights(model_pretrained.get_layers(l.name).get_weight())
#                     freeze_all(l)
#     else:
#         model.load_weights(weights)
#         if transfer == 'fine_tune':
#             darknet = model.get_layer('yolo_darknet')
#             freeze_all(darknet)
#         elif transfer == 'frozen':
#             freeze_all(model)
        
#     return model


# # 测试检测图片
# yolov3_weights = "C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/checkpoints/yolov3.tf"

# model = YoloV3(size=IMG_SIZE, train=False, classes=num_classes)
# model.load_weights(yolov3_weights)

# img_raw = tf.image.decode_image(open('C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/meme.jpg', 'rb').read(), channels=3)
# img = tf.expand_dims(img_raw, 0)
# img = transform_image(img=img, img_size=IMG_SIZE)

# boxes, scores, classes, nums = model(img)
# img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
# img = cv2.resize(img, dsize=(512, 512))

# class_names = [c.strip() for c in open('C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/coco.names').readlines()]
# output_img = draw_output_img(img, (boxes, scores, classes, nums), class_names)

# cv2.imshow('show_img', output_img)
# cv2.waitKey(0)

# # 使用 tf.GradientTape 梯度计算  (失败：跑不满CPU，无法利用GPU全部性能)
# # tf.compat.v1.disable_eager_execution()
# iterator = get_tfrecord_iterator("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
# model = YoloV3(size=IMG_SIZE, train=True)

# opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # model.compile(optimizer=opt, loss=loss)


# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     # sess.run(iterator.initializer)
#     one_batch = iterator.get_next()
#     all_loss = 0
#     pos = 0
#     model.summary()
#     for epoch in range(epochs):
#         for batch_pos in tqdm(range(batch_nums)):
#             train_x, train_y = sess.run(one_batch)
#             with tf.GradientTape() as tape:
#                 pred = model(train_x)
#                 loss_fns = [YoloLoss(anchors=yolo_anchors[mask], classes=num_classes) for mask in yolo_anchor_masks]
#                 regularization_loss = tf.reduce_sum(model.losses)
#                 loss = []
#                 loss.append(loss_fns[0](train_y[0], pred[0]))
#                 loss.append(loss_fns[1](train_y[1], pred[1]))
#                 loss.append(loss_fns[2](train_y[2], pred[2]))
#                 fina_loss = tf.reduce_sum(loss) + regularization_loss
#                 all_loss += fina_loss
#                 pos+=1
#             grad = tape.gradient(fina_loss, model.trainable_variables)
#             opt.apply_gradients(zip(grad, model.trainable_variables))
#             # print("loss=", sess.run(fina_loss))


# 使用Model.fit()
tf.compat.v1.disable_eager_execution()
model = YoloV3(IMG_SIZE, train=True, classes=num_classes)
yolov3_weights = "C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/checkpoints/yolov3.tf"
model.load_weights(yolov3_weights)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
loss = [YoloLoss(yolo_anchors[mask], classes=num_classes)
        for mask in yolo_anchor_masks]

model.compile(optimizer=optimizer, loss=loss,
                run_eagerly=False)
train_batch = get_tfrecord_iterator("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
callbacks = [
    ReduceLROnPlateau(verbose=1),
    EarlyStopping(patience=3, verbose=1),
    ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                    verbose=1, save_weights_only=True),
    TensorBoard(log_dir='logs')
]

start_time = time.time()
history = model.fit(train_batch,
                    epochs=epochs,
                    callbacks=callbacks)
end_time = time.time() - start_time
print(f'Total Training Time: {end_time}')
