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


def create_img_path_idx(save_len = 1000, shuffer = True):
    path = "D:/IDM/dog_cat/train"
    file_path = "./yolov3_tf2/test/data/cat_and_dog.txt"
    file = open(file_path, 'w')
    imgs = os.listdir(path)
    img_list = []
    for img in tqdm(imgs):
        img_path = os.path.join(path, img)
        img_path = img_path + " " + ('0' if img.startswith('cat') else '1') + '\n'
        img_list.append(img_path)
        # file.write(img_path)
    if shuffer:
        random.shuffle(img_list)
    for tmp in tqdm(img_list[0:save_len]):
        file.write(tmp)
    file.close()

def parse_xml_info_voc(label_path):
    dom_xml = parse(label_path)
    xml_data = dom_xml.documentElement
    # annotation = xml_data.getElementsByTagName('filename')[0].childNodes[0].data
    # print(annotation)
    plt.title(xml_data.getElementsByTagName('filename')[0].childNodes[0].data)
    w= xml_data.getElementsByTagName("size")[0].getElementsByTagName("width")[0].childNodes[0].data
    h = xml_data.getElementsByTagName("size")[0].getElementsByTagName("height")[0].childNodes[0].data
    objs = xml_data.getElementsByTagName("object")
    boxs = []
    for obj in objs:
        box = [0] * 5
        box[4] = obj.getElementsByTagName("name")[0].childNodes[0].data
        bboxs = obj.getElementsByTagName('bndbox')
        box[0] = bboxs[0].getElementsByTagName('xmin')[0].childNodes[0].data
        box[1] = bboxs[0].getElementsByTagName('ymin')[0].childNodes[0].data
        box[2] = bboxs[0].getElementsByTagName('xmax')[0].childNodes[0].data
        box[3] = bboxs[0].getElementsByTagName('ymax')[0].childNodes[0].data
        box[0] = (float(box[0]) / float(w))
        box[1] = (float(box[1]) / float(h))
        box[2] = (float(box[2]) / float(w))
        box[3] = (float(box[3]) / float(h))
        boxs.append(box)
    return boxs


def draw_bbox_with_float(img_path, boxs, img_size=128):
    colors = ['blue', 'black', 'red', 'green', 'pink', 'yellow', 'orange']

    img_raw = Image.open(img_path)
    img_raw = img_raw.resize((img_size, img_size))
    plt.imshow(img_raw)
    plt.title(img_path[max(img_path.rfind('/')+1, img_path.rfind('\\')+1):])
    ax = plt.gca()

    # 画框与标签
    for bboxs in boxs:
        bboxs[0:4] = [x*img_size for x in bboxs[0:4]]
        tmp_color = colors[random.randint(0, 6)]
        font = FontProperties(size=12)
        ax.text(bboxs[0], bboxs[1], bboxs[4], fontproperties=font, bbox={'alpha':0.4, 'facecolor':tmp_color})
        rect = patcher.Rectangle((bboxs[0], bboxs[1]), (bboxs[2]-bboxs[0]), (bboxs[3]-bboxs[1]), edgecolor=tmp_color, fill=False)
        ax.add_patch(rect)

    plt.show()


def draw_rect_voc(pos = 0):
    IMG_SIZE = 128
    colors = ['blue', 'black', 'red', 'green', 'pink', 'yellow', 'orange']
    voc_img_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\JPEGImages"
    voc_label_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\Annotations"

    # 取出目标文件
    img_list = os.listdir(voc_img_path)
    img_name = img_list[pos]

    # 得到对应的图片与标签
    img_abs_name = img_name[0: img_name.rfind('.')]
    label_name = img_abs_name + ".xml"
    img_name = os.path.join(voc_img_path, img_name)
    label_name = os.path.join(voc_label_path, label_name)
    # print(img_name)
    # print(label_name)

    img_raw = Image.open(img_name)
    img_raw = img_raw.resize((IMG_SIZE, IMG_SIZE))
    plt.imshow(img_raw)
    ax = plt.gca()

    # lxml错误解析。。。
    # label_raw = xml.fromstring(open(label_name).read())
    
    # annotation = (label_raw)['annotation']
    # plt.title(annotation['filename'])
    # w, h = annotation['size']['width', 'height']
    # for boxs in annotation['object']:
    #     type, xmin, ymin, xmax, ymax = boxs['name'], boxs['bndbox']['xmin', 'ymin', 'xmax', 'ymax']
    #     xmin, ymin, xmax, ymax = float(xmin) / w, float(ymin) / h, float(xmax) / w, float(ymax) / h
    #     tmp_color = colors[random.randint(0, 6)]
    #     ax.add_patch((xmin * IMG_SIZE, ymin * IMG_SIZE), (xmax-xmin)*IMG_SIZE, (ymax-ymin)*IMG_SIZE, color=tmp_color, fill=False)
    #     ax.text((xmin * IMG_SIZE, ymin * IMG_SIZE), type, bbox={'facecolor':tmp_color})

    # xml.dom.minidom 解析
    dom_xml = parse(label_name)
    xml_data = dom_xml.documentElement
    # annotation = xml_data.getElementsByTagName('filename')[0].childNodes[0].data
    # print(annotation)
    plt.title(xml_data.getElementsByTagName('filename')[0].childNodes[0].data)
    w= xml_data.getElementsByTagName("size")[0].getElementsByTagName("width")[0].childNodes[0].data
    h = xml_data.getElementsByTagName("size")[0].getElementsByTagName("height")[0].childNodes[0].data
    objs = xml_data.getElementsByTagName("object")
    for obj in objs:
        type = obj.getElementsByTagName("name")[0].childNodes[0].data
        box = [0] * 20
        bboxs = obj.getElementsByTagName('bndbox')
        box[0] = bboxs[0].getElementsByTagName('xmin')[0].childNodes[0].data
        box[1] = bboxs[0].getElementsByTagName('ymin')[0].childNodes[0].data
        box[2] = bboxs[0].getElementsByTagName('xmax')[0].childNodes[0].data
        box[3] = bboxs[0].getElementsByTagName('ymax')[0].childNodes[0].data
        box[0] = int(float(box[0]) / float(w) * IMG_SIZE)
        box[1] = int(float(box[1]) / float(h) * IMG_SIZE)
        box[2] = int(float(box[2]) / float(w) * IMG_SIZE)
        box[3] = int(float(box[3]) / float(h) * IMG_SIZE)

        # 画框
        tmp_color = colors[random.randint(0, 6)]
        rect = patcher.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), edgecolor=tmp_color, fill=False)
        ax.add_patch(rect)
        ax.text(box[0], box[1], s=type, bbox={'facecolor':tmp_color, 'alpha':0.3})

        # print(type, box)
    plt.show()

def test_get_voc_info():
    pos = 1
    voc_img_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\JPEGImages"
    voc_label_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\Annotations"

    # 取出目标文件
    img_list = os.listdir(voc_img_path)
    img_name = img_list[pos]

    # 得到标签路径
    img_abs_name = img_name[0: img_name.rfind('.')]
    label_name = img_abs_name + ".xml"
    label_name = os.path.join(voc_label_path, label_name)
    bboxs = parse_xml_info_voc(label_name)
    print(bboxs)

def test_draw_box_with_float(pos=0):
    voc_img_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\JPEGImages"
    voc_label_path = "C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\data\\voc2012_raw\\VOCdevkit\\VOC2012\\Annotations"

    # 取出目标文件
    img_list = os.listdir(voc_img_path)
    img_name = img_list[pos]

    # 得到对应的图片与标签
    img_abs_name = img_name[0: img_name.rfind('.')]
    label_name = img_abs_name + ".xml"
    img_name = os.path.join(voc_img_path, img_name)
    label_name = os.path.join(voc_label_path, label_name)

    draw_bbox_with_float(img_name, parse_xml_info_voc(label_name), 256)

def get_max_boxes():
    path = "D:\\IDM\\yolo\\face_mask\\dataset\\images\\train"
    list = os.listdir(path)

    nums = [0] * 50
    for it in list:
        if not it.endswith('txt'):
            continue
        file_name = os.path.join(path, it)
        print(file_name)
        file = open(file_name, 'r')
        lines = file.readlines()
        pos = min(len(lines), 49)
        nums[pos] = nums[pos] + 1
    print(nums)

import cv2

def face_mask_data_proc():
    classes = ['mask', 'nomask']
    path = "D:\\IDM\\yolo\\face_mask\\dataset\\images\\train"
    save_dir = "data"
    save_name = "face_mask.tfrecords"
    save_path = os.path.join(path, save_dir)
    save_name = os.path.join(save_path, save_name)

    writer = tf.io.TFRecordWriter(save_name)

    tmp_img_list = os.listdir(path)
    img_list = []
    for it in tmp_img_list:
        if(it.endswith('jpg')):
            img_list.append(it)
    for it in tqdm(img_list):
        label_name = it[0:it.rfind('.')] + ".txt"
        img_name = os.path.join(path, it)
        label_all_name = os.path.join(path, label_name)
        img_raw = Image.open(img_name)
        img_raw = img_raw.resize((128, 128))
        weight, hight = img_raw.size
        img_data = img_raw.tobytes()
        # label_list = []
        label_info = open(label_all_name, 'r')
        objs = label_info.readlines()
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        labels = []
        labels_text = []
        for obj in (objs):
            label, xmin, ymin, w, h = obj.split(' ')
            xmin = float(xmin)
            ymin = float(ymin)
            ymax = ymin + float(h)
            xmax = xmin + float(w)
            # print([label, xmin, ymin, xmax, ymax])
            xmins.append((xmin))
            ymins.append((ymin))
            xmaxs.append((xmax))
            ymaxs.append((ymax))
            labels_text.append(classes[int(label)].encode('utf-8'))
            labels.append(int(label))
            # label_list.append([label_name, label, xmin, ymin, xmax, ymax])
        label_info.close()

        # print(label_list)

        # c = input()

        example = tf.train.Example(
            features=tf.train.Features( feature={
                'image/filename' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[it.encode('utf-8')])),
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
                'image/weight' : tf.train.Feature(int64_list=tf.train.Int64List(value=[weight])),
                'image/height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[hight])),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
                'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=labels_text)),
                'image/object/class/label' : tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
            })
        )
        writer.write(example.SerializeToString())
    writer.close()

def face_mask_read_data(single_example):
    classes = ['mask', 'nomask']
    path = "D:\\IDM\\yolo\\face_mask\\dataset\\images\\train"
    save_dir = "data"
    save_name = "face_mask.tfrecords"
    save_path = os.path.join(path, save_dir)
    save_name = os.path.join(save_path, save_name)

    EXAMPLE_MAP = {
        'image/encoded' : tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin' : tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin' : tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax' : tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax' : tf.io.VarLenFeature(tf.float32),
        'image/object/class/text' : tf.io.VarLenFeature(tf.string),
    }

    data = tf.io.parse_single_example(single_example, EXAMPLE_MAP)
    # x_train = data['image/encoded']
    # x_train = tf.io.decode_jpeg(x_train, channels=3)

    # y_text = data['image/object/class/text']
    y_min = data['image/object/bbox/ymin']
    print(y_min)
    # plt.imshow(x_train)
    # plt.show()

def face_mask_test():
    files = tf.data.Dataset.list_files("C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/voc2012_val.tfrecord")                    # 返回file_pattern可以匹配到的所有图片路径
    dataset = files.flat_map(tf.data.TFRecordDataset)                   # 打开图片
    dataset.map(lambda x: face_mask_read_data(x))

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
def parse_tfrecord(tfrecord):
    size = 128
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)     # 处理单条Example的函数，多条用parse_example
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)  # 解析图片
    x_train = tf.image.resize(x_train, (size, size))
    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     plt.imshow(x_train.eval())
    #     plt.show()
    # print(x_train)


def face_mask_test2():
    files = tf.data.Dataset.list_files("C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/voc2012_val.tfrecord")                    # 返回file_pattern可以匹配到的所有图片路径
    dataset = files.flat_map(tf.data.TFRecordDataset)                   # 打开图片
    dataset.map(lambda x: face_mask_read_data(x))

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


YOLO_MAX_BOXES = 20
YOLO_IMG_SIZE = 128
yolo_epoch = 1
yolo_batch_size = 1

def get_tfrecord_iterator(tfrecord_file_path, shuffle=True, epoch=yolo_epoch, batch_size=yolo_batch_size):
    def _parse_example_(example):
        features = tf.io.parse_single_example(example, features={
            'image/filename' : tf.io.VarLenFeature(tf.string),
            'image/weight' : tf.io.FixedLenFeature((1), tf.int64),
            'image/height' : tf.io.FixedLenFeature((1), tf.int64),
            'image/encoded' : tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.float32),
        })
        w = features['image/weight'][0]
        h = features['image/height'][0]
        img_raw = features['image/encoded']
        '''
            解析图片
            img_raw = open(img_path, 'rb').read() / img_raw = tf.image.encode_jpeg()
                    |
                    |
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)  # 解析图片
            x_train = tf.image.resize(x_train, (size, size))


            img_raw = Image.open(img_name)
            img_raw = img_raw.resize((128, 128))
            img_data = img_raw.tobytes()
                    |
                    |
            img = tf.io.decode_raw(img_raw, tf.uint8) 
            img = tf.compat.v1.reshape(img, (w, h, 3))

        '''
        img = tf.io.decode_raw(img_raw, tf.uint8)       # 解析图片
        img = tf.compat.v1.reshape(img, (w, h, 3))
        labels = features['image/object/class/label']
        xmins = features['image/object/bbox/xmin']
        ymins = features['image/object/bbox/ymin']
        xmaxs = features['image/object/bbox/xmax']
        ymaxs = features['image/object/bbox/ymax']
        boxs = tf.stack([tf.sparse.to_dense(xmins),         # [batch, xmin]
                          tf.sparse.to_dense(ymins),        # [batch, ymin]
                          tf.sparse.to_dense(xmaxs),        # [batch, xmax]         ---->      [batch, (xmin, ymin, xmax, ymax, label)]
                          tf.sparse.to_dense(ymaxs),        # [batch, ymax]
                          tf.sparse.to_dense(labels)]       # [batch, label]
                         , axis=1) 
        
        # 规整形状为一致
        pad_len = YOLO_MAX_BOXES - tf.shape(boxs)[0]
        if pad_len > 0: boxs = tf.pad(boxs, [[0, pad_len], [0, 0]])
        else: boxs = boxs[0:YOLO_MAX_BOXES]
        return img, boxs

    def transform_image(img, img_size=YOLO_IMG_SIZE):
        img = tf.image.resize(img, (img_size, img_size))
        img = img / 255.0
        return img

    
    # 每次处理一张图片的数据
    def transform_label(label):     # [[[0.065  0.70139396  0.186  1.0051358  1.  5.], [0.42225  0.06382979  0.46275  0.1298606  0.  1.], ... ]]]

        def pre_trans_label(label, grid_size, anchor_idxs):
            # [grid_size, grid_size, anchors_size, (xmin, ymin, xmax, ymax, obj, anchor_idx)]
            y_true_out = tf.zeros([grid_size, grid_size, tf.shape(anchor_idxs)[0], 6])

            anchor_idxs = tf.cast(anchor_idxs, tf.int32)

            indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
            updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
            idx = 0

            # return label[0]

            for j in tf.range(tf.shape(label)[0]):
                if tf.equal(label[j][2], 0):     # 空框
                    continue

                anchor_equ = tf.equal(anchor_idxs, tf.cast(label[j][5], tf.int32))        # 最合适的框是否在这一批次
                
                if not tf.reduce_any(anchor_equ):       # 不在这一批次
                    continue

                box = label[:, 0:4]
                center_xy = (box[j][0:2] + box[j][2:4]) / 2     # 中心点xy
                grid_xy = tf.cast(center_xy // (1./grid_size), tf.int32)                # 在哪个grid中
                anchor_idx = tf.cast(tf.where(anchor_equ), tf.int32)                    # [0, 1, 4, 0, 0] -->  [0,1],[0,2]  哪个坐标的数值为非零（可以看最合适的框是否在这一批次中）

                indexes = indexes.write(idx, [grid_xy[1], grid_xy[0], anchor_idx[0][0]])                        # 替换的坐标[第几行的grid中， 第几列的grid中， 用的第几个bound_box]
                updates = updates.write(idx, [box[j][0], box[j][1], box[j][2], box[j][3], 1, label[j][4]])      # 替换的内容[xmin, ymin, xmax, ymax, 1, label]
                idx += 1
            # tf.tensor_scatter_nd_update:(稀疏矩阵表示形式)，将y_true_out的indexes位置的点用updates的内容替换
            # [grid_size, grid_size, anchors_size, (xmin, ymin, xmax, ymax, obj, anchor_idx)]
            return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


        grid_size = YOLO_IMG_SIZE // 32
        y_outs = []

        anchors = tf.cast(yolo_anchors, tf.float32)
        wh = label[..., 2:4] - label[..., 0:2]                  # [[[0.12100001 0.3037418 ] [0.04049999 0.06603081], ... ]]
        wh = tf.expand_dims(wh, -2)                             # [[ [[0.12100001 0.3037418 ]] [[0.04049999 0.06603081]], ... ]]
        # wh = tf.tile(wh, (1, 1, 1))   # (将某一维度的数据赋值n次) 将第三维的数据复制tf.shape(anchor)[0]次
        
        # 查找与真实框最契合的bndbox
        box_area = wh[..., 0] * wh[..., 1]                      # [[[0.03675276], [0.00267425] ... ]]
        anchor_area = anchors[..., 0] * anchors[..., 1]         # [[ 0.0007512  0.00277367  0.00438586 0.01057461 0.01612195 0.04057068 0.06032729 0.17848557 0.7026512]]
        intersection = tf.minimum(wh[..., 0], anchors[..., 0]) * tf.minimum(anchors[..., 1], wh[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)        # [[[5.], [1.], ... ]]
        
        label = tf.concat([label, anchor_idx], axis=-1)         # [[[0.065  0.70139396  0.186  1.0051358  1.  5.], [0.42225  0.06382979  0.46275  0.1298606  0.  1.], ... ]]]
        
        for anchor_idxs in yolo_anchor_masks:
            y_outs.append(pre_trans_label(label, grid_size, anchor_idxs))
            grid_size *= 2
        return tuple(y_outs)
    
    tf.compat.v1.disable_eager_execution()
    dataset = tf.compat.v1.data.TFRecordDataset(tfrecord_file_path)
    dataset = dataset.map(_parse_example_)
    dataset = dataset.map(lambda x, y:(
        transform_image(x),
        transform_label(y)
    ))
    dataset = dataset.shuffle(shuffle).repeat(epoch).batch(batch_size)

    # train_iterator = dataset.make_one_shot_iterator()
    # return train_iterator
    return dataset

def face_mask_test_3(tfrecord_path="C:/Users/LJ/Desktop/tensorflow_code/yolo-v5/yolo-v3/yolov3-tf2/data/voc2012_val.tfrecord"):
    train_iterator = get_tfrecord_iterator(tfrecord_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_batch = train_iterator.get_next()

        for i in range(1000):
            train_x, train_y = sess.run(train_batch)

            print(train_y)
            plt.imshow(train_x[0])
            # plt.show()
            input()



if __name__ == '__main__':
    # create_img_path_idx()

    # draw_rect_voc(1)

    # get_voc_info()

    # for i in range(10):
    #     test_draw_box_with_float(random.randint(0, 100))
    # get_max_boxes()
    # face_mask_data_proc()
    # face_mask_read_data()
    face_mask_test_3("D:/IDM/yolo/face_mask/dataset/images/train/data/face_mask.tfrecords")
    pass