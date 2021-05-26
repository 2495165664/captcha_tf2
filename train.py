import make_data
from network import my_network
from tensorflow.keras import losses, optimizers


train_images, train_labels = make_data.make()
# model = my_network.mySequential((60, 120, 1), (4, 10))
# scope = model(train_images[:16])
# print(scope)
# loss = losses.MeanAbsoluteError()
# loss_da = loss(train_labels[:16], scope)
#
# print(loss_da)

def run():
    # model = my_network.mySequential((60, 120, 1), (4, 10))
    # model.summary()
    model = my_network.MyModel(output=(4, 10))
    model.build(input_shape=(16, 60, 120, 1))
    # opti = optimizers.Adam(lr=0.00001)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=500, batch_size=16)
    model.save('./model/exp1')


run()