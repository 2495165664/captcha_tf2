import random
import os
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha  # pip install captcha

from settings_tf import config


# 验证码保存路径
SAVE_PATH = './data/images/'
# label txt 路径
LABEL_TXT_PATH = './data/label.txt'
# 图片长度
IMAGE_WIDTH = 120
# 图片宽度
IMAGE_HEIGHT = 60
# 生成数量
create_num = 100
# 生成字符集
setts = config.number + config.alphabet
# 字符集长度
captcha_size = 4


if os.path.exists(SAVE_PATH) == False:
    os.mkdir(SAVE_PATH)

# if os.path.exists(LABEL_TXT_PATH) == False:
#     os.mkdir(LABEL_TXT_PATH)

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=setts, captcha_size=captcha_size):
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
    create(create_num)
