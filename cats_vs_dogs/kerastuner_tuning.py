from kerastuner import RandomSearch
from numpy import load
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow_core.python.keras.preprocessing.image import ImageDataGenerator

from cats_vs_dogs.cats_vs_dogs import define_three_block_model, define_one_block_model


# TODO: Not working so far, look at talos


def tune_with_kerastuner1():
    photos = load('dogs_vs_cats_photos.npy')
    labels = load('dogs_vs_cats_labels.npy')
    (trainX, testX, trainY, testY) = train_test_split(photos, labels, test_size=0.25, random_state=42)

    trainY = keras.utils.to_categorical(trainY, 2)
    testY = keras.utils.to_categorical(testY, 2)

    model = define_three_block_model()

    # history = model.fit(photos, labels, batch_size=16, epochs=10, validation_split=0.33, verbose=1, use_multiprocessing=True)

    tuner = RandomSearch(model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory="tuner_dir",
                         project_name="cats_vs_dogs_tuner")
    tuner.search_space_summary()

    # tuner.search(trainX, trainY,
    #              epochs=5,
    #              validation_data=(testX, testY))

    models = tuner.get_best_models(num_models=2)
    tuner.results_summary()
    return tuner


def tune_with_kerastuner2():
    # model = define_one_block_model()
    tuner = RandomSearch(define_one_block_model, objective='val_accuracy', max_trials=5, executions_per_trial=3,
                         directory="tuner_dir",
                         project_name="cats_vs_dogs_tuner")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                 class_mode='binary', batch_size=16, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                               class_mode='binary', batch_size=16, target_size=(200, 200))

    tuner.search(train_it, steps_per_epoch=len(train_it),
                 validation_data=test_it, validation_steps=len(test_it), epochs=5,
                 use_multiprocessing=True)

    models = tuner.get_best_models(num_models=3)
    tuner.results_summary()
    return tuner
