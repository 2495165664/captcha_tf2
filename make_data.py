import os

import cv2
import numpy as np
import tensorflow as tf

from tools import progress_bar
from settings_tf import config
# one = tf.one_hot(5, depth=10)
# print(one)


def get_images_path():
    """
    获取所有图片路径
    :return: [...]
    """
    paths_name = os.listdir(config.IMAGE_PATH)
    # 重新排序
    paths_name.sort(key=lambda x: int(x[:-4]))

    return paths_name


def read_image(image_path):
    """
    读取图片
    :param image_path:
    :return:
    """
    image = cv2.imread(os.path.join(config.IMAGE_PATH, image_path))
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
    images_data = np.ndarray(shape=((len(images_path),) + config.IMAGE_SIZE), dtype='uint8')
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
    f = open(config.IMAGE_LAEBL_PATH, 'r')
    data = f.readlines()
    labels_data = np.ndarray(shape=((len(data),) + config.LAEBL_SIZE), dtype='int')
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
