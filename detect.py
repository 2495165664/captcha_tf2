import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import models

from tools import detect_data

# 测试图片路径
image_path = './data/test_images'
# 模型加载路径
model_path = './model/exp1'
# 结果存放路径 保存格式 path + date .txt
save_path = 'data/detect/'
# 加载图片和图片名
images, images_name = detect_data.get_data(image_path)

# 是否存在保存路径
if os.path.exists(save_path) == False:
    os.mkdir(save_path)


def load_model():
    # 加载模型
    model = models.load_model(model_path)
    model.summary()
    return model

def run():
    # 加载模型
    model = load_model()

    # 模型预测
    pred = model(images)

    path = save_path + str(datetime.now())[8:16].replace(' ', '-').replace(":", '-') + '.txt'
    print(path)
    f = open(path, 'w', encoding='utf-8')

    for index, i in enumerate(pred):
        pred_value = []
        print(images_name[index])
        for j in i:
            pred_ = tf.argmax(j, axis=0)
            pred_value.append(pred_.numpy())
        f.write(str(pred_value) + '    ')
        f.write(images_name[index] + "\n")

        print(pred_value)
        print("*" * 50)

    f.close()

try:
    run()
    print("Sussessful!")
except Exception as e:
    print("Error! 报错:", e)