from tensorflow.keras import models, layers, optimizers
import make_data
import make_test_data


MAX_CAPTCHA = 4
CHAR_SET_LEN = 10


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(60, 160, 1)),
    layers.MaxPooling2D((2, 2), strides=2),

    layers.Conv2D(64, (5, 5), activation="relu"),
    layers.MaxPooling2D((2, 2), strides=2),
    layers.Dropout(0.2),
    layers.Conv2D(128, (5, 5), activation="relu"),
    layers.MaxPooling2D((2, 2), strides=2),

    layers.Flatten(),
    layers.Dropout(0.3),
    # layers.Dense(1024, activation='relu'),
    layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN, activation='relu'),
    layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN, activation='relu'),
    layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN, activation='softmax'),
    layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN])
])

model.summary()
model.compile(optimizer="Adam",
              metrics=['accuracy'],
              loss='categorical_crossentropy')

model.load_weights("model_test.h5")


for i in range(10000):
    print("*"*80, "- >>  ", i)
    train_images, train_labels = make_train_data.load_data()
    test_images, test_labels = make_test_data.load_data()
    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save("model_test", save_format="tf")
    break
    # model.save_weights("model_test.h5")
