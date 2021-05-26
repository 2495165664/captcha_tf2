from tensorflow.keras import models
import tensorflow as tf
import cv2


from tools import detect_data


images = detect_data.get_data('./data/test_images')

def load_model():
    # 加载模型
    model = models.load_model('./model/exp1')
    model.summary()
    return model

# 加载模型
model = load_model()

pred = model(images[0:3])
# for image in images:
#     # 预测结果
#     predict =model(image.reshape((1,) + image.shape))
#     pred_value = []
#     for i in predict[0]:
#         value = tf.argmax(i, axis=0)
#         pred_value.append(value.numpy())
#         # print(value)
#     print(pred_value)
for i in pred:
    for j in i:
        pred_ = tf.argmax(j, axis=0)
        print(pred_)
    print("*"*50)
    print(pred.shape)
    print(i.shape)


cv2.imshow("a", images[0])
cv2.waitKey(0)
