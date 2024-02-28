import tensorflow as tf
from absl.flags import FLAGS

def transform_targets_for_output(y_true, grid_size, anchor_idxs):   # [batch_size, boxes, [xmin, ymin, xmax, ymax, label, anchor_idx](百分比格式)]  --> [batch_size, grid, grid, anchor_nums, [x1, y1, x2, y2, obj, class]]
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):                                               # 第几张图片
        for j in tf.range(tf.shape(y_true)[1]):                         # 第几个框
            if tf.equal(y_true[i][j][2], 0):                                # xmax vs 0     ??
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))            # [6,7,8] vs 0-8 ([6,7,8] == 7 -->  [0,1,0])

            if tf.reduce_any(anchor_eq):                                    # [0,1,0]   -->  true;      [0,0,0] -> false
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2        # 中心点坐标(百分比)

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)         # [0,1,0]    ->  1
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)        # 在第几个格子(x, y)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])     # 第几个有效框， [第几张图片, 第几行的grid中， 第几列的grid中， 用的第几个bound_box]
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])      # 第几个有效框， [xmin, ymin, xmax, ymax, 1, label]
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    # tf.tensor_scatter_nd_update:(稀疏矩阵表示形式)，将y_true_out的indexes位置的点用updates的内容替换
    return tf.tensor_scatter_nd_update(                                     # 填充y_true_output
        y_true_out, indexes.stack(), updates.stack())

'''
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

'''

def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # 看这个box与那个bound_box最为契合
    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)     # [batch_size, [xmin, ymin, xmax, ymax, label, anchor_idx]]

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2
    # [batch_size, grid, grid, anchor_nums, [x1, y1, x2, y2, obj, class]]
    return tuple(y_outs)


# 处理图片（归一化，resize）
def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
# 解析tfrecord的规则集合
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


# 从tfrecord文件中读取img, label数据，并转换成需要的形状
def parse_tfrecord(tfrecord, class_table, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)     # 处理单条Example的函数，多条用parse_example
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)  # 解析图片
    x_train = tf.image.resize(x_train, (size, size))

    # space_to_dense(data, output_shape, value, default_value):稀疏矩阵转化成稠密矩阵（制作one_hot矩阵）
    # tf.space.to_dense(): ???
    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')         # x['text']未知 （可能是：[x, y](矩阵形状) + [...](n个数据), [[., .] * n(n个数据的位置)]
    labels = tf.cast(class_table.lookup(class_text), tf.float32)        # text --> text_id([batch, label])
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),        # [batch, xmin]
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),        # [batch, ymin]
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),        # [batch, xmax]         ---->>      [batch, [xmin, ymin, xmax, ymax, label]]
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),        # [batch, ymax]
                        labels], axis=1)                                        # [batch, label]

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train

# 将file_pattern可以匹配到的路径打开后经过处理后成为数据集返回
def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(      # TF内置字典（初始化：将文件中的所有类别(以delimiter为分隔符)从1-n标号）
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)                    # 返回file_pattern可以匹配到的所有图片路径
    # apply：将数据交给函数， 返回处理后的结果      x([..]) --f-->  y[...]
    # map：将数据依次交给函数，将结果合成原来结构并返回   [x1, x2, ...xn]  --f-->  [y1, y2, ...yn]
    # flat_map：将数据展平后交给函数处理， 返回处理后的结果（flat+apply）       x1, x2, ..xn -> X[x1, x2, ..xn] -f-> Y[...]
    dataset = files.flat_map(tf.data.TFRecordDataset)                   # 打开图片
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))  # 对图片进行处理


# 生成一张图片构成的训练集
def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
