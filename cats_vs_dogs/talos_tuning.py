import numpy as np
import talos
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

from util import get_train_and_test_sets, tfinit


def tune_with_talos():
    # train_it, test_it = get_train_and_test_sets()

    p = {
        'activation': ['relu', 'elu'],
        'optimizer': ['Nadam', 'Adam'],
        'loss': ['binary_crossentropy'],
        'batch_size': [16],
        'epochs': [5, 10, 15],
        'target_size': [(200, 200), (100, 100), (64, 64)],  # original 200x200
        'kernel_size': [(3, 3), (4, 4), (5, 5)]  # orignal 3x3
    }

    def three_block_model(x, y, x_val, y_val, params):
        target_size = params['target_size']

        model = Sequential()
        model.add(Conv2D(32, kernel_size=params['kernel_size'], activation=params['activation'], kernel_initializer='he_uniform',
                         padding='same',
                         input_shape=(target_size[0], target_size[1], 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, kernel_size=params['kernel_size'], activation=params['activation'], kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, kernel_size=params['kernel_size'], activation=params['activation'], kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=params['activation'], kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        # opt = SGD(lr=0.001, momentum=0.9)

        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        # width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        # prepare iterators

        train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                     class_mode='binary', batch_size=16, target_size=target_size)
        test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                                   class_mode='binary', batch_size=16, target_size=target_size)

        model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'])

        out = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=params['epochs'],
                                  use_multiprocessing=True)
        return out, model

    dummy_x = np.empty((1, 10, 3, 200, 200))
    dummy_y = np.empty((1, 10))

    scan_object = talos.Scan(dummy_x, dummy_y, model=three_block_model, params=p, experiment_name="cats_vs_dogs talos results")
    # fraction_limit=0.1)
    return scan_object


def test_model():
    train_it, test_it = get_train_and_test_sets()

    params = {
        'activation': 'relu',
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'batch_size': 16,
        'epochs': 5,
        'target_size': (200, 200),
        'kernel_size': (3, 3)
    }

    def three_block_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation=params['activation'], kernel_initializer='he_uniform', padding='same',
                         input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation=params['activation'], kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation=params['activation'], kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=params['activation'], kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        # opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'])

        out = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=10,
                                  use_multiprocessing=True)
        return out, model

    return three_block_model()


def minimal():
    x, y = talos.templates.datasets.iris()

    p = {'activation': ['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'losses': ['binary_crossentropy'],
         'batch_size': [20, 30, 40],
         'epochs': [10, 20]}

    def iris_model(x_train, y_train, x_val, y_val, params):
        model = Sequential()
        model.add(Dense(32, input_dim=4, activation=params['activation']))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=params['optimizer'], loss=params['losses'])

        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=[x_val, y_val],
                        verbose=0)

        return out, model

    scan_object = talos.Scan(x, y, model=iris_model, params=p, experiment_name='iris', fraction_limit=0.1)

    return scan_object


tfinit()
# scan_obj = minimal()
scan_obj = tune_with_talos()

# history, model = test_model()
