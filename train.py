import make_data
from network import my_network
from tensorflow.keras import losses, optimizers


train_images, train_labels = make_data.make()


def run():
    model = my_network.MyModel(output=(4, 10))
    model.build(input_shape=(16, 60, 120, 1))
    # opti = optimizers.Adam(lr=0.00001)
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=2, batch_size=16)
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    print(accuracy)
    print(loss)
    # print(history)
    model.save('./model/exp1')


run()