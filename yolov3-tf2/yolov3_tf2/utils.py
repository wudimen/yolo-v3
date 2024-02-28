from absl import logging
import numpy as np
import tensorflow as tf
import cv2

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


# 根据结构依次层级遍历 ， 取出并设置数据
def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')       # 打开权重文件
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    # YOLO模型结构
    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)         # 取出model中与结构名字对应的层级
        for i, layer in enumerate(sub_model.layers):        #遍历层级中的各层并取出对应数据并设置
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):     # 是否使用batch_norm
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters         # 卷积核数量
            size = layer.kernel_size[0]     # 卷积核大小
            in_dim = layer.get_input_shape_at(0)[-1]


            # 没用batch_norm就有bias，反之亦成
            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)    # 取偏置量的数据
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(                                       # 取batch_norm的数据
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)                          # 权重的形状
            conv_weights = np.fromfile(                                         # 取出权重数据
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])                    # 设置权重与偏置量的值
            else:
                layer.set_weights([conv_weights])                               # 设置权重与batch_norm的值
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


# 根据预测结果与原始标签计算IOU
def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)       # (..., (x1, y1, x2, y2))->(..., add, (x1, y1, x2, y2))
    box_2 = tf.expand_dims(box_2, 0)        # (N, (x1, y1, x2, y2))->(add, N, (x1, y1, x2, y2))
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -       # 较小的x2 - 较大的x1 -> 重叠面积的宽
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -       # 较小的y2 - 较大的y1 -> 重叠面积的高
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)      # 返回IOU

# 用输出结果给图片标框
# img:图片， outputs：输出结果[batch_size, {bbox{x, y, w, h}*N, 有物体的概率*N, 类别ID*N， 预测框数量N}], 类别合集
def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

# 用原始标签给图片标框
#   x:图片， y:标签[batch_size, 5{classes, x, y, w, h}], 类别集合
def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)       # bbox-[batch_size, 4], 类别-[batch_size, 1]
    classes = classes[..., 0]       # 类别-[batch_size]
    wh = np.flip(img.shape[0:2])    # 从图片的shape中获取宽高-[batch_size, 2]
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))       # 真实左上角坐标
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))       # 真是右下角坐标
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)        # 画框
        img = cv2.putText(img, class_names[classes[i]],             # 标注类别
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


# 递归使model里的层级都不训练（model.trainable = false）
def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
