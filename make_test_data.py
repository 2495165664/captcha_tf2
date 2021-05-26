import os
import numpy as np
import cv2

PATH = "./test_images/"


number = ['0','1','2','3','4','5','6','7','8','9']
MAX_CAPTCHA = 4
CHAR_SET_LEN = len(number)


def get_images_path():
    paths_name = os.listdir(PATH)
    return paths_name

def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = number.index(c)
        vector[i][idx] = 1.0
    return vector


def parse_data(paths):
    train_images = np.ndarray(shape=(len(paths), 60, 160, 1), dtype="float32")
    train_labels = np.zeros([len(paths), MAX_CAPTCHA, CHAR_SET_LEN], dtype="uint8")
    temp = 0
    for i in paths:
        path = PATH + i
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        train_labels[temp, :] = text2vec(i[:4])
        train_images[temp] = array.reshape(60, 160, 1)
        temp += 1

    return train_images, train_labels

def load_data():
    # create_image.create_num(1000)
    paths_name = get_images_path()
    return parse_data(paths_name)


if __name__ == '__main__':
    paths_name = get_images_path()
    train_images, train_labels = parse_data(paths_name)
    cv2.imshow("a", train_images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()