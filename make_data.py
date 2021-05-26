import os

import cv2
import numpy as np
import tensorflow as tf

from tools import progress_bar

# one = tf.one_hot(5, depth=10)
# print(one)

# 数据集存放路径
IMAGE_PATH = "./data/test_images/"
# label存放路径
IMAGE_LAEBL_PATH = './data/test_label.txt'
# 声明图片长宽和图层
IMAGE_SIZE = (60, 120, 1)
# 输出层， 例如4个数字[4, 10]
LAEBL_SIZE = (4, 10)


def get_images_path():
    """
    获取所有图片路径
    :return: [...]
    """
    paths_name = os.listdir(IMAGE_PATH)
    return paths_name


def read_image(image_path):
    """
    读取图片
    :param image_path:
    :return:
    """
    image = cv2.imread(os.path.join(IMAGE_PATH, image_path))
    # image转灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    return image


def imageData_to_num():
    """
    所有图片放进一个多维数组
    :return:
    """
    images_path = get_images_path()
    # print(images_path)
    images_data = np.ndarray(shape=((len(images_path),) + IMAGE_SIZE), dtype='uint8')
    long = len(images_data)
    # print("正在加载图片: ", end='')
    for i, image_path in enumerate(images_path):
        image = read_image(image_path)
        images_data[i, :, :, 0] = image
        progress_bar.run(i, long, '>>> images 加载')
    print()
    print("images data size", images_data.shape)
    return images_data


def lable_to_num():
    """
    所有label放进数组
    :return: [n, 4, 10]
    """
    f = open(IMAGE_LAEBL_PATH, 'r')
    data = f.readlines()
    labels_data = np.ndarray(shape=((len(data),) + LAEBL_SIZE), dtype='int')
    long = len(labels_data)
    for o, i in enumerate(data):
        i = i.replace("\n", '')
        for v, j in enumerate(i):
            labels_data[o, v] = tf.one_hot(int(j), depth=10).numpy()

        progress_bar.run(o, long, ">>> label 加载")
    print()
    print("labels size:", labels_data.shape)
    f.close()
    return labels_data


def make():
    """
    [] []
    :return:
    """
    return imageData_to_num(), lable_to_num()
