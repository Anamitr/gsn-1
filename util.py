import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def tfinit():
    # Magic line to prevent some cuda error on my computer
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def get_train_and_test_sets(target_size=(200, 200)):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                 class_mode='binary', batch_size=16, target_size=target_size)
    test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                               class_mode='binary', batch_size=16, target_size=target_size)
    return train_it, test_it
