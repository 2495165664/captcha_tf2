import os
import cv2
import numpy as np

# 自写文件
from settings_tf import config


def get_data(imagepath):
    """
    返回 (b, long, width, 3)
    :param imagepath:图片路径
    :return:
    """
    paths = os.listdir(imagepath)
    paths.sort(key=lambda x: int(x[:-4]))
    # print(paths)
    # 
    images_data = np.ndarray(shape=((len(paths),) + config.IMAGE_SIZE), dtype='uint8')
    for i, path in enumerate(paths):
        print(path)
        path = os.path.join(imagepath, path)
        # 读取图片
        image = cv2.imread(path)
        # 灰度处理并归一化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_data[i, :, :, 0] = image
            # print(image)
            # break
    print("test image:", images_data.shape)
    return images_data


images = get_data('../data/images')
# print(images.shape)