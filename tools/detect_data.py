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
    images_name = []
    for i, path in enumerate(paths):
        images_name.append(path)

        path = os.path.join(imagepath, path)
        # 读取图片
        image = cv2.imread(path)
        # 灰度处理并归一化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 固定大小
        image = cv2.resize(image, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        images_data[i, :, :, 0] = image
            # print(image)
            # break
    print("test image:", images_data.shape)
    return images_data, images_name


# images, a = get_data('../data/test_images')
#
# cv2.imshow('1', images[0])
# cv2.waitKey(0)