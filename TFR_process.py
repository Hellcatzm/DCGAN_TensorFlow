# Author : hellcat
# Time   : 18-1-21

import os
import glob
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

np.set_printoptions(threshold=np.inf)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 读取数据文件的轮数
NUM_EPOCHS = None  # 不限制，否则会报out_of_range_error

# TFR保存图像尺寸
IMAGE_HEIGHT = 48
IMAGE_WIDTH = IMAGE_HEIGHT
IMAGE_DEPTH = 3
# 训练batch尺寸
BATCH_SIZE = 64
# 定义每个TFR文件中放入多少条数据
INSTANCES_PER_SHARD = 10000
# 图片文件存放路径
IMAGE_PATH = './Data_Set/cartoon_faces'
# 图片文件和标签清单保存文件
IMAGE_LABEL_LIST = 'images_&_labels.txt'
# TFR文件保存路径
TFR_PATH = './TFRecord_Output'


def filename_list(class_num='0', path=IMAGE_PATH):
    """
    文件清单生成
    :param class_num:类别名，str类型
    :param path:图像路径，path下直接是图片
    :return: txt文件，每一行内容是：路径图片名+若干空格+类别标签数字+\n
    """
    # 获取图片名称以及数量
    # 等价于image_names = glob.glob(path+'/*')
    # 使用next可以直接取出迭代器中的元素
    try:
        file_names = next(os.walk(path))[2]
    except IOError:
        print('文件列表为空')

    with open(IMAGE_LABEL_LIST, 'w') as f:
        for file_name in file_names:
            f.write(path+'/'+file_name+' '+class_num+'\n')


def image_to_tfr(image_and_label=IMAGE_LABEL_LIST,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
    """
    从清单读取图片并生成TFR文件
    :param image_and_label: txt图片清单
    :param image_height: 保存如TFR文件的图片高度
    :param image_width: 保存TFR文件的图片宽度
    """

    def _int64_feature(value):
        """生成整数数据属性"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        """生成字符型数据属性"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    if not os.path.exists('./TFRecord_Output'):
        os.mkdir('TFRecord_Output')
    with open(image_and_label, 'r') as f:
        lines = f.readlines()
        image_paths = [image_path.strip('\n').split(' ')[0] for image_path in lines]
        labels = [image_path.strip('\n').split(' ')[-1] for image_path in lines]

        # 如下操作会报错，因为忽略了指针问题，第一次readlines后指针到达文件末尾，第二次readlines什么都read不到
        # image_paths = [image_path.strip('\n').split(' ')[0] for image_path in f.readlines()]
        # labels = [image_path.strip('\n').split(' ')[-1] for image_path in f.readlines()]

    num_file = len(image_paths)
    # 定义写多少个文件(数据量大时可以写入多个文件加速)
    num_shards = num_file // INSTANCES_PER_SHARD + 1

    for file_i in range(num_shards):
        # 文件名命名规则
        file_name = os.path.join(TFR_PATH, '{0}.tfrecords_{1}_of_{2}') \
            .format(image_paths[0].split('/')[-2], file_i + 1, num_shards)
        print('正在生成文件: ', file_name)
        # 书写器初始化
        writer = tf.python_io.TFRecordWriter(file_name)
        for index, image_path in enumerate(
                image_paths[file_i * INSTANCES_PER_SHARD:(file_i + 1) * INSTANCES_PER_SHARD]):
            image_data = imread(os.path.join(image_path))
            image_data = imresize(image_data, (image_height, image_width))
            image_raw = image_data.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image_raw),
                'label': _int64_feature(int(labels[index]))
            }))
            writer.write(example.SerializeToString())
        # 书写器关闭
        writer.close()


def batch_from_tfr(image_height=IMAGE_HEIGHT,
                   image_width=IMAGE_WIDTH,
                   image_depth=IMAGE_DEPTH):
    """从TFR文件读取batch数据"""

    if not os.path.exists(TFR_PATH):
        os.makedirs(TFR_PATH)

    '''读取TFR数据并还原为uint8的图片'''
    file_names = glob.glob(os.path.join(TFR_PATH, '{0}.tfrecords_*_of_*')
                           .format(IMAGE_PATH.split('/')[-1]))
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=NUM_EPOCHS, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image = features['image']
    image_decode = tf.decode_raw(image, tf.uint8)
    # 解码会变为一维数组，所以这里设定shape时需要设定为一维数组
    image_decode.set_shape([image_height * image_width * image_depth])
    image_decode = tf.reshape(image_decode, [image_height, image_width, image_depth])
    label = tf.cast(features['label'], tf.int32)

    '''图像预处理'''
    image_decode = tf.cast(image_decode, tf.float32)/127.5-1

    '''生成batch图像'''
    # 随机获得batch_size大小的图像和label
    images, labels = tf.train.shuffle_batch([image_decode, label],
                                            batch_size=BATCH_SIZE,
                                            num_threads=1,
                                            capacity=10000 + 3 * BATCH_SIZE,  # 队列最大容量
                                            min_after_dequeue=1000)
    return images, labels
    # # 测试部分
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # img = sess.run(images)[0]
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # coord.request_stop()
    # coord.join(threads)


if __name__ == '__main__':
    import datetime

    time1 = datetime.datetime.now()
    filename_list()
    image_to_tfr()
    # batch_from_tfr()
    time2 = datetime.datetime.now()
    print(time2 - time1)