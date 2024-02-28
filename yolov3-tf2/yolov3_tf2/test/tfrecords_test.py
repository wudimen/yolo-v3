# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys

tfrecords_path = "./yolov3_tf2/test/data/"

# tfrecords初体验
def test_1():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'
    with tf.io.TFRecordWriter(path=tfrecords_file) as writer:
        writer.write(b'123')
        writer.write(b'xyz123')
        writer.close()

    with open(tfrecords_file, 'rb') as file:
        print(file.read())

# tf张量与string的转换
def test_2():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.uint8)       # tf张量
    print(x)
    x = tf.io.serialize_tensor(x)                           #tf张量-->string
    print(x)
    x = tf.io.parse_tensor(x, out_type=tf.uint8)            #string-->tf张量
    print(x)

# tfrecords存储string类型
def test_3():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'
    
    # data = tf.constant([[1, 2], [3, 4]], dtype=tf.uint8)

    data = tf.data.Dataset.from_tensor_slices([b'123', b'abc123'])      # 数据集
    writer = tf.data.experimental.TFRecordWriter(tfrecords_file)        # 写入tfrecords
    writer.write(data)

    data_2 = tf.data.TFRecordDataset(tfrecords_file)                    # 读取tfrecords
    for ds in data_2:
        print(ds)

# tfrecords存储其他类型的张量（存储时转换成string，读取时还原）
def test_4():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'
    
    features = tf.constant([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8 ,9]], dtype=tf.uint8)
    ds = tf.data.Dataset.from_tensor_slices(features)       # 张量数据
    ds_bytes = ds.map(tf.io.serialize_tensor)               # 张量数据  ->  string

    writer = tf.data.experimental.TFRecordWriter(tfrecords_file)
    writer.write(ds_bytes)                             # 存储转换后的张量

    ds_byte_2 = tf.data.TFRecordDataset(tfrecords_file)
    # 1.使用lambda                                          # 读取tfrecords后转换成原来的格式
    # ds_2 = ds_byte_2.map(lambda x:tf.io.parse_tensor(x, out_type=tf.uint8))

    # 2.自定义函数
    def _parse_uint8(x):
        return tf.io.parse_tensor(x, out_type=tf.uint8)
    ds_2 = ds_byte_2.map(_parse_uint8)


    for data in ds_2:
        print(data)

from sklearn.datasets import load_sample_image

# 图像的编码与解码
def test_decode_img():
    img_path = "d:/image/1.jpg"
    # img_raw = load_sample_image("C:\\Users\\LJ\\Desktop\\tensorflow_code\\yolo-v5\\yolo-v3\\yolov3-tf2\\yolov3_tf2\\test\\1.jpg")
    img_raw = tf.compat.v1.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw)
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # plt.imshow(img.numpy())
    # tf.compat.v1.initialize_variables(tf.compat.v1.global_variables_initializer)
    with tf.compat.v1.Session() as sess:
        img = sess.run(img)
        plt.imshow(tf.image.decode_jpeg(img))
        plt.title("img")
        plt.show()
    
    cv2.waitKey(0)


# 图像序列化
def test_5():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'

# 复合数据储存
    
def make_example(img, label, class_id):
    features = tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(img).numpy()])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'class_id' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_id.encode()])),
    })
    return tf.train.Example(features=features).SerializeToString()

def test_6():
    print(make_example(np.array([[1, 2], [3, 4], [5, 6]]), 1, "person"));



################################################################################

# string / bytes
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf8') if type(value)==str else value]))

# float32 / double(float64)
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# int / uint / enum / bool
def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# int list
def _int_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# byte list(数组)
def _byte_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def create_example_test(feature0, feature1, feature2, feature3, feature4):
    feature = {
        'feature0' : _int_feature(feature0),
        'feature1' : _int_list_feature(feature1),
        'feature2' : _bytes_feature(feature2),
        'feature3' : _float_feature(feature3),
        'feature4' : _byte_list_feature(feature4)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature = feature))
    print(example_proto)
    return example_proto.SerializeToString()

def test_2_1_write():
    tfrecords_file = tfrecords_path + 'test_' + str(sys._getframe().f_code.co_name) + '.tfrecords'
    
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for i in range(10):
            example = create_example_test(1, [1,2,3], "hhh", 1.23, np.array(['11','1','311']));
            writer.write(example)

def test_2_1_read():
    feature_description = {
        'feature0' : tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'feature1' : tf.io.FixedLenFeature((3), tf.int64, default_value=[-1, -1, -1]),
        'feature2' : tf.io.FixedLenFeature((), tf.string, default_value=''),
        'feature3' : tf.io.FixedLenFeature((), tf.float32, default_value=0.0),
        'feature4' :  tf.io.FixedLenFeature((), tf.string)
    }
    def _pause_function(example_proto):
        feature = tf.io.parse_single_example(example_proto, feature_description)
        feature['feature4'] = tf.compat.v1.decode_raw(feature['feature4'], out_type=tf.float32)
        return feature

    tf.compat.v1.disable_eager_execution()
    tmp_path = tfrecords_path+"test_test_2_1_write.tfrecords"
    data_set = tf.data.TFRecordDataset(tmp_path) # .compat.v1.data.TFRecordDataset(tfrecords_path+"tfrecords_test_2_1_write.records")
    print(tmp_path)
    data_set = data_set.map(_pause_function)
    # print(data_set)
    data_set = data_set.batch(1)
    # iterator = data_set.make_one_shot_iterator()

    iterator = tf.compat.v1.data.make_one_shot_iterator(data_set)
    data = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        datas = []
        for i in range(1):
            my_data = sess.run([data])
            datas.append(my_data)
        print(datas)

import random
import os
from PIL import Image
from tqdm import tqdm
##########################################################################################
def create_record(record_path, img_dir, img_txt):
    writer = tf.io.TFRecordWriter(record_path)

    # 读取img_txt 文件
    img_list = []
    with open(img_txt) as file:
        img_list = file.readlines()
    # print(img_list)
    random.shuffle(img_list)

    for img_info in tqdm(img_list):
        img_name = img_info.split(' ')[0]       # 图片路径
        img_cls = int(img_info.split(' ')[1])        # 图片类别

        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        # print(img_name)
        # print(img)

        #处理图像
        img = img.resize((128, 128))
        img_raw = img.tobytes()

        # 声明要储存的内容
        example = tf.train.Example(
            features = tf.train.Features(feature={
                'label' : tf.train.Feature(int64_list = tf.train.Int64List(value=[img_cls])),
                'img_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })
        )
        writer.write(example.SerializeToString())
    writer.close()

# import tensorflow as tf

def parse_record(record_path):
    def _parse_func(example_proto):
        features = tf.compat.v1.parse_single_example(example_proto, features={
            'label' : tf.compat.v1.FixedLenFeature([], tf.compat.v1.int64),
            'img_raw' : tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)
        })
        label = features['label']
        img_raw = features['img_raw']
        img = tf.compat.v1.decode_raw(img_raw, tf.compat.v1.uint8)

        # 处理图片
        img = tf.compat.v1.reshape(img, (128, 128, 3))
        # img = tf.compat.v1.cast(img, tf.compat.v1.float32) * (1 / 255.0) - 0.5
        label = tf.compat.v1.cast(label, tf.compat.v1.int32)
        return img, label
    
    def _data_iterator(tfrecords):
        tf.compat.v1.disable_eager_execution()      # 关闭eager模式（tf2.x默认开启， 要使用tf.session的话就不是eager模式，需关闭）

        # 声明TFRecordDataset
        dataset = tf.compat.v1.data.TFRecordDataset(tfrecords)
        dataset = dataset.map(_parse_func)
        # 打乱顺序，设置epoch，batch_size
        dataset = dataset.shuffle(True).repeat(1).batch(1)
        # 定义iterator
        iterator = dataset.make_one_shot_iterator()
        return iterator
    
    train_iterator = _data_iterator(record_path)

    # print(record_path)
    # dataset = tf.compat.v1.data.TFRecordDataset(record_path)
    # dataset = dataset.map(_parse_func)
    # dataset = dataset.shuffle(buffer_size = 1000).repeat(1).batch(1)
    # train_iterator = dataset.make_one_shot_iterator()

    # tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.initialize_all_variables().run()
        train_batch = train_iterator.get_next()
        for i in tqdm(range(1000)):
            train_x, train_y = sess.run(train_batch)
            # if i%10 != 0:
            #     continue
            # print(train_y)
            # plt.imshow(train_x[0])
            # plt.show()


if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_decode_img()
    # test_6()

    ##################################################################
    # test_2_1_write()
    test_2_1_read()

    ##################################################################
    # create_record(tfrecords_path+'cat_and_dog.tfrecords', '', tfrecords_path+'cat_and_dog.txt')
    # parse_record(tfrecords_path+'cat_and_dog.tfrecords')