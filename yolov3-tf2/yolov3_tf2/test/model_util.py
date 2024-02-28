import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Add,
    Concatenate,
    Input,
    Lambda,
    MaxPool2D,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

YOLO_MAX_BOXES = 30
YOLO_IOU_THRESHOLD = 0.5
YOLO_SCORE_THRESHOld = 0.5

# DarkNet卷积层
def DarkNetConv(x, filters, size, strides=1, Batchnorm=True):
    if strides==1:
        pad='same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        pad='valid'
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=pad, use_bias=not Batchnorm, kernel_regularizer=l2(0.0005))(x)

    if Batchnorm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


# DarkNet残差网络（与前面的网络数值相加，以达到网络层数变多效率至少不会变低的效果）
def DarkNetResidual(x, filters):
    prev = x
    x = DarkNetConv(x, filters // 2, 1)
    x = DarkNetConv(x, filters, 3)
    x = Add()([x, prev])
    return x

# DarkNet网络块
def DarkNetBlock(x, filters, count):
    x = DarkNetConv(x, filters, 3, strides=2)
    for i in range(count):
        x = DarkNetResidual(x, filters)
    return x

# DarkNet网络模型
def DarkNet(name=None):
    x = input = Input([None, None, 3])
    x = DarkNetConv(x, 32, 3)
    x = DarkNetBlock(x, 64, 1)
    x = DarkNetBlock(x, 128, 2)
    x = x_36 = DarkNetBlock(x, 256, 8)
    x = x_61 = DarkNetBlock(x, 512, 8)
    x = DarkNetBlock(x, 1024, 4)
    return Model(input, (x_36, x_61, x), name=name)

# Yolo卷积层集
def YoloConvSet(filters, name=None):
    def yolo_conv(x_in):        # 是否是残差网络
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = DarkNetConv(x, filters, 1)
            x= UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
        
        x = DarkNetConv(x, filters, 1)
        x = DarkNetConv(x, filters * 2, 3)
        x = DarkNetConv(x, filters, 1)
        x = DarkNetConv(x, filters * 2, 3)
        x = DarkNetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


# Yolo输出层
def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = input = Input(x_in.shape[1:])
        x = DarkNetConv(x, filters * 2, 3)
        x = DarkNetConv(x, anchors * (classes + 5), 1, Batchnorm=False)
        x = Lambda(lambda x:tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes+5)))(x)
        return Model(input, x, name=name)(x_in)
    return yolo_output

#
def _meshgrid(m_a, m_b):        # (3, 2)    -->     [ [[0,1,2],[0,1,2]], [[0,0,0],[1,1,1]] ]
    return [
        tf.reshape(tf.tile(tf.range(m_a), [m_b]), [m_b, m_a]),
        tf.reshape(tf.repeat(tf.range(m_b), m_a), [m_b, m_a])
    ]


# 将相对于对应格子相对坐标的box 转换成 相对于整张图片的相对坐标的box ， 处理并取出obj,classes
def yolo_pred_to_boxes(pred, anchors, classes):     # pred:[batch_size, grid_size, grid_size, anchors, (c_x, c_y, w, h, obj, classes)]
    grid = tf.shape(pred)[1:3]     # 获得grid大小
    c_xy, x_wh, obj, classs = tf.split(pred, (2, 2, 1, classes), axis=-1)   # 分开中心点x,y,宽，高，有无obj，类别

    # 处理一下pred数据，使之更容易收敛
    c_xy = tf.sigmoid(c_xy)
    classs = tf.sigmoid(classs)
    obj = tf.sigmoid(obj)
    pred_box = tf.concat([c_xy, x_wh], axis=-1)

    # 生成一个数组：x,y分别从0-grid[0]与0-grid[1]
    grid = _meshgrid(grid[1], grid[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    # 每个格子的坐标xy分别加上对应格子的下标，（从对于格子的相对坐标转化从相对于整张图片的相对坐标）
    box_xy = (c_xy + tf.cast(grid, tf.float32)) / tf.cast(grid, tf.float32)  # 相对整张图片的相对坐标
    box_wh = tf.exp(x_wh) * anchors       # wh经过e^x处理，更容易收敛

    # 左上角/右下角坐标
    box_xy1 = box_xy - box_wh / 2
    box_xy2 = box_xy + box_wh / 2

    bbox = tf.concat([box_xy1, box_xy2], axis=-1)

    # 处理后的box坐标，有无物体，类别，未处理的box坐标
    return bbox, obj, classs, pred_box


# 非极大值抑制处理输出的labels
def yolo_nms(output, classes):
    # 取出数据并摊平数据[batch_size, grid, grid, anchors, (box/conf/classes)]  ->  [batch_size, -1, (box/conf/classes)]
    box, conf, classs = [], [], []
    for o in output:
        box.append(tf.reshape(o[0], [tf.shape(o[0])[0], -1, tf.shape(o[0])[-1]]))
        conf.append(tf.reshape(o[1], [tf.shape(o[1])[0], -1, tf.shape(o[1])[-1]]))
        classs.append(tf.reshape(o[2], [tf.shape(o[2])[0], -1, tf.shape(o[2])[-1]]))
    
    # 将第三维的数据整合在一起（减少了一维）
    box = tf.concat(box, axis=1)
    conf = tf.concat(conf, axis=1)
    classs = tf.concat(classs, axis=1)

    if classes == 1:
        score = conf
    else :
        score = conf * classs
    
    dscore = tf.squeeze(score, axis=0)      # 取出维度数为一的维度
    score = tf.reduce_max(dscore, [1])       # 分数最高的值
    classes = tf.argmax(dscore, axis=1)     # 分数最高的下标
    box = tf.reshape(box, [-1, 4])          # [batch_size, 4]

    # 非最大值抑制
    # 参数：框的坐标[batch_size, (xmin ymin xmax ymax)]
    #       对应框的分数[batch_size, (score)]
    #       最多从中选择多少个框
    #       与分数最高的框IOU值高于这个值时，舍去这个框
    #       得分低于这个值时，删除这个框
    #       ？
    # 返回值：返回选中框的下标(eg:[0,2,3..])
    selected_idx, selected_score = tf.image.non_max_suppression_with_scores(
        boxes=box,
        scores=score,
        max_output_size=YOLO_MAX_BOXES,
        iou_threshold=YOLO_IOU_THRESHOLD,
        score_threshold=YOLO_SCORE_THRESHOld,
        soft_nms_sigma=0.5
    )

    num_valid_boxes = tf.shape(selected_idx)[0]     # 选择的候选框数量
    
    # 将selected_idx, selected_score 规整至合适大小
    selected_idx = tf.concat([selected_idx, tf.zeros(YOLO_MAX_BOXES-num_valid_boxes, tf.int32)], 0)
    selected_score = tf.concat([selected_score, tf.zeros(YOLO_MAX_BOXES-num_valid_boxes, tf.float32)], -1)

    box = tf.gather(box, selected_idx)      # 将选中框对应的boxes抽取出来
    box = tf.expand_dims(box, axis=0)
    score = selected_score                  # 选择后的score
    score = tf.expand_dims(score, axis=0)
    classes = tf.gather(classes, selected_idx)     # 将选中框对应的类别抽取出来
    classes = tf.expand_dims(classes, axis=0)
    num_valid_boxes = tf.expand_dims(num_valid_boxes, axis=0)

    # 经过NMS得到的boxes坐标，分数，类别，以及其中有效框的数量
    return box, score, classes, num_valid_boxes

import numpy as np
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def YoloV3(size=None, channels=3, anchors=yolo_anchors, anchor_masks=yolo_anchor_masks, classes=80, train=False):
    x = input = Input([size, size, channels])
    x_36, x_61, x = DarkNet(name='yolo_darknet')(x)

    x = YoloConvSet(filters=512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(filters=512, anchors=len(anchor_masks[0]), classes=classes, name='yolo_output_0')(x)

    x = YoloConvSet(filters=256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(filters=256, anchors=len(anchor_masks[1]), classes=classes, name='yolo_output_1')(x)

    x = YoloConvSet(filters=128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(filters=128, anchors=len(anchor_masks[2]), classes=classes, name='yolo_output_2')(x)

    if train:
        return Model(input, (output_0, output_1, output_2), name='yolov3')
    
    boxes_0 = Lambda(lambda x:yolo_pred_to_boxes(x, anchors[anchor_masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x:yolo_pred_to_boxes(x, anchors[anchor_masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x:yolo_pred_to_boxes(x, anchors[anchor_masks[2]], classes), name="yolo_boxes_2")(output_2)

    output = Lambda(lambda x:yolo_nms(x, classes), name='yolo_output')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    return Model(input, output, name='yolov3')


def caculate_iou(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=-2)
    y_pred = tf.expand_dims(y_pred, axis=0)

    new_shape = tf.broadcast_dynamic_shape(y_true, y_pred)
    y_true = tf.broadcast_to(y_true, new_shape)
    y_pred = tf.broadcast_to(y_pred, new_shape)

    interuct_w = tf.maximum(tf.minimum(y_true[..., 2], y_pred[..., 2]) - tf.maximum(y_true[..., 0], y_pred[..., 0]), 0)
    interuct_h = tf.maximum(tf.minimum(y_true[..., 3], y_pred[..., 3]) - tf.maximum(y_true[..., 1], y_pred[..., 1]), 0)
    interuct_ares = interuct_w * interuct_h

    true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
    pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])

    iou = interuct_ares / (true_area + pred_area - interuct_ares)
    return iou

def YoloLoss(anchors, classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # y_true:   [batch_size, grid, grid, anchors, (xmin, ymin, xmax, ymax, obj, classes)]
        # y_pred:   [batch_size, grid, grid, anchors, (cent_x, cent_y, w, h, obj, classes)]

        # y_pred info:
        pred_box, pred_obj, pred_classes, pred_xywh = yolo_pred_to_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # y_true info
        true_box, true_obj, true_classes = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2]  + true_box[..., 2:4]) / 2
        true_wh = (true_box[..., 2:4] - true_box[..., 0:2])

        # 给不同大小的框给与不同的权重（小框的权重值更大，防止忽略小框以获取更大的loss值）
        score_boxes = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 调整true_xy, true_wh的格式，使之与pred对应
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.shape(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)      # 从相对一张图片的位置变成相对对应格子的相对位置
        true_wh = tf.math.log(true_wh / anchors)        # 之前为了更好传递true_wh的误差，e ^ wh， 此时恢复过来
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)        # 经过tf.math.log，结果可能为inf，这里将inf变成0，防止后续计算loss值时'loss爆炸'

        # 是否有真实物体（只计算真实物体的box_loss与class_loss）
        obj_mask = tf.squeeze(true_obj, axis=-1)
        best_iou = tf.map_fn(lambda x:tf.reduce_max(caculate_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1)(pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 计算各个loss值
        xy_loss = obj_mask * score_boxes * tf.reduce_sum(tf.square(true_xy-pred_xy), axis=-1)
        wh_loss = obj_mask * score_boxes * tf.reduce_sum(tf.square(true_wh-pred_wh), axis=-1)
        obj_loss = tf.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1-obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * tf.losses.sparse_categorical_crossentropy(true_classes, pred_classes)       # TODO 使用binary_crossentropy

        # 求和全部loss值
        xy_loss = tf.reduce_sum(xy_loss, axis=(1,2,3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1,2,3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1,2,3))
        class_loss = tf.reduce_sum(class_loss, axis=(1,2,3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss

if __name__ == '__main__':
    model = YoloV3(size=128, train=False)
    # model = DarkNet(name='darknet')
    model.summary()