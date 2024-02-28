import logging
import os
from tqdm import tqdm
import random
from PIL import Image
import lxml.etree as xml
import matplotlib.pyplot as plt
import matplotlib.patches as patcher
from matplotlib.font_manager import FontProperties
from xml.dom.minidom import parse
import numpy as np

import tensorflow as tf

def test_1():
    x = tf.concat([1,3,5,7,9,0], axis=0)
    y = tf.concat(range(0, len(x)), axis=0)
    print(x, y)
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    z = tf.concat([y, x], 1)
    print(x, y, z)
    z = tf.compat.v1.sparse_to_dense(z, [len(x), 10], 1.0, 0.0)
    print(z)
    # print(tf.concat([tf.expand_dims([1,2,3], 1), tf.expand_dims([1,2], 1)], axis=0))

def test_2():
    sp_input = tf.SparseTensor(
    dense_shape=[3, 5],
    values=[7, 8, 9, 1],
    indices =[[0, 1],
                [0, 3],
                [2, 0],
                [2, 1]])
    data = tf.sparse.to_dense(sp_input).numpy()
    print(data)

def test_3():
    # 创建一个 4x4 的张量
    tensor = tf.zeros(shape=[4, 4], dtype=tf.float32)

    # 定义要更新的位置和值
    indices = [[0, 0], [1, 2], [3, 1]]  # 更新位置的索引
    updates = [1.0, 2.0, 3.0]  # 更新位置的值

    # 使用 tf.tensor_scatter_nd_update 函数更新张量
    updated_tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)

    print(updated_tensor)

def test_4():

    # labels = []
    # labels.append([[1,2,3], [1,2,3], [1,2,3]])
    # labels.append([2,3])
    # print(labels)
    print(tf.where([[0, 3, 5]]))

from util import get_tfrecord_iterator
from model_util import (
    YoloV3,
    YoloLoss,
)

epochs = 1
batch_num = 50
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

tf.compat.v1.disable_eager_execution()
model = YoloV3(size=128)
# 优化器
optimizer = tf.keras.optimizers.Adam(lr=0.001)
# 损失值
loss = [YoloLoss(yolo_anchors[mask], classes=80) for mask in yolo_anchor_masks]

# 优化
model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)

# avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
# avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

def train():
    train_dataset = get_tfrecord_iterator("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
    callbacks = [
        ReduceLROnPlateau(verbose=1),               # 降低学习率条件
        EarlyStopping(patience=3, verbose=1),       # 提前结束条件
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',      # 存储模型设置
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')             # 训练过程显式化
    ]

    history = model.fit(train_dataset,      # 训练模型
                        epochs=1,
                        callbacks=callbacks)

    # iterator = get_tfrecord_iterator("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
    # train_data = iterator.get_next()
    # with tf.compat.v1.Session() as sess, tf.GradientTape() as tape:
    #     for epoch in range(epochs):
    #         for batch_pos in tqdm(range(batch_num)):
    #             train_x, train_y = sess.run(train_data)
    #             pred = model(train_x, training=True)
    #             regularization_loss = tf.reduce_sum(model.losses)
    #             pred_loss = []
    #             for output, label, loss_fn in zip(pred, train_y, loss):
    #                 pred_loss.append(loss_fn(label, output))
    #             total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    #         grads = tape.gradient(total_loss, model.trainable_variables)
    #         optimizer.apply_gradients(
    #             zip(grads, model.trainable_variables))
    #         logging.info("{}_train_{}, {}, {}".format(
    #             epoch, batch_pos, total_loss.numpy(),
    #             list(map(lambda x: np.sum(x.numpy()), pred_loss))))
    #         avg_loss.update_state(total_loss)


if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_4()
    train()
