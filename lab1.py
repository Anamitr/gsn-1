from __future__ import absolute_import, division, print_function, unicode_literals, print_function

from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

fashion_mnist = keras.datasets.fashion_mnist

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ręczne uruchomienie predykcji
predictions = model.predict(x_test)
print("Przewidywana cyfra to:", np.argmax(predictions[0]), "; prawdziwa cyfra:", np.argmax(y_test[0]))


# Testing
SAVE_NAME = 'fashion_mnist_model'

model.save(SAVE_NAME + '.h5')
# model = keras.models.load_model(SAVE_NAME + '.h5')

predictions = model.predict(x_test)
num = 0
print("Przewidywana cyfra to:", np.argmax(predictions[num]), "; prawdziwa cyfra:", (y_test[num]))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def show_img(image, label, size=1):
    plt.figure(figsize=(size, size))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    plt.xlabel(label)
    plt.show()


num = 0

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# show_img(x_test[num], class_names[y_test[num]])

# Show image and predicted label
show_img(x_test[num], class_names[np.argmax(predictions[num])])

