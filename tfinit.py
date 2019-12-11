import tensorflow as tf


def tfinit():
    # Magic line to prevent some cuda error on my computer
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
