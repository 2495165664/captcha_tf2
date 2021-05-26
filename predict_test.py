import tensorflow as tf
from tensorflow.keras import models, layers
import make_test_data


MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


model = models.load_model("./model_test")
test_images, test_labels = make_test_data.load_data()

model.fit(test_images, test_labels, verbose=2, epochs=1)

print(model.get_weights())
