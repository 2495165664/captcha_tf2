import random

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha  # pip install captcha

# 验证码保存路径
SAVE_PATH = './data/images/'
# label txt 路径
LABEL_TXT_PATH = './data/label.txt'
# 图片长度
IMAGE_WIDTH = 120
# 图片宽度
IMAGE_HEIGHT = 60

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(num):
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    image.write(captcha_text, SAVE_PATH + str(num) + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def create(num):
    f = open(LABEL_TXT_PATH, 'w')

    for i in range(num):
        text, img = gen_captcha_text_and_image(i+1)
        print(i, text)
        f.write(text + '\n')
    f.close()

if __name__ == '__main__':
    create_num = 5000
    create(create_num)
