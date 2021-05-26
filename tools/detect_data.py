import os
import cv2


def get_data(imagepath):
    """
    返回 (b, long, width, 3)
    :param imagepath:图片路径
    :return:
    """
    paths = os.listdir(imagepath)
    print(paths)
    # 
    images_data = np.ndarray(shape=((len(paths),) + IMAGE_SIZE), dtype='uint8')
    for path in paths:
        path = os.path.join(imagepath, path)
        image = cv2.imread(path)
        print(image)
        break


get_data('../data/test_images')