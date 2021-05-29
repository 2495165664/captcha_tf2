import make_data
from network import my_network
from settings_tf import config
import os

# 指定GPU训练
os.environ['CUDA_VISIBLE_DEVICES']='2, 3, 4'
train_images, train_labels = make_data.make()

# 训练次数
epochs = 100
# 分批大小
batch_size = 16

def run():
    model = my_network.MyModel(output=config.LABEL_SIZE)
    model.build(input_shape=((batch_size,) + config.IMAGE_SIZE ))
    # opti = optimizers.Adam(lr=0.00001)
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=config.train_and_val)
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    with open('./result.txt', 'w')as f:
        f.write(str(accuracy))
        f.write('\n')
        f.write(str(loss))
        f.write('\n')
        f.write(str(history.history['val_loss']))
        f.write('\n')
        f.write(str(history.history['val_accuracy']))
    # print(history)
    model.save('./model/exp1')


run()
