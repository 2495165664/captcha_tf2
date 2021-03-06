from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Reshape, Softmax, Dropout, BatchNormalization


class MyModel(Model):
    def __init__(self, output):
        super(MyModel, self).__init__()
        # 归一化，防止梯度爆炸
        self.make = layers.experimental.preprocessing.Rescaling(1. / 255)
        # 第一层卷积 1 > 16
        self.conv1 = Conv2D(32, 2, activation='relu')
        self.batch1 = BatchNormalization()

        self.conv2 = Conv2D(64, 2, activation='relu')
        self.batch2 = BatchNormalization()
        self.maxpadding2 = MaxPooling2D((2, 2), padding='same')


        self.conv3 = Conv2D(128, 1, activation='relu')
        self.batch3 = BatchNormalization()
        self.maxpadding3 = MaxPooling2D((2, 2), padding='same')

        self.conv6 = Conv2D(256, 3, activation='relu')
        self.batch6 = BatchNormalization()
        self.maxpadding6 = MaxPooling2D((2, 2), padding='same')

        self.conv4 = Conv2D(64, 2, activation='relu')
        self.batch4 = BatchNormalization()
        self.maxpadding4 = MaxPooling2D((2, 2), padding='same')

        self.conv5 = Conv2D(32, 2, activation='relu')
        self.batch5 = BatchNormalization()
        self.maxpadding5 = MaxPooling2D((2, 2), padding='same')

        self.flatten = Flatten()
        self.d1 = Dense(output[0] * output[1], activation='softmax')
        self.out = Reshape(output)

    def call(self, x, training=None, mask=None):
        x = self.make(x)

        x = self.conv1(x)
        x = self.batch1(x)
        # x = self.maxpadding1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.maxpadding2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.maxpadding3(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.maxpadding6(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.maxpadding4(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.maxpadding5(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.out(x)

        return x

# model = MyModel(output=(4, 10))
# model.build(input_shape=(16, 60, 120, 1))
# model.summary()

