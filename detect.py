from tensorflow.keras import models
import tensorflow as tf
import make_data
import cv2


images, laebls = make_data.make()


model = models.load_model('./model/exp1')
model.summary()

pred = model(images[0: 1])

for i in pred:
    for j in i:
        pred_ = tf.argmax(j, axis=0)
        print(pred_)
    print(pred.shape)
    # print(i.shape)

cv2.imshow("a", images[2])
cv2.waitKey(0)