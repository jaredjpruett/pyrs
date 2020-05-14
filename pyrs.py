from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model, model_from_json
from keras import backend
import numpy as np
import json


def load_images(images):
    imgs = []

    for im in images:
        img = image.load_img(im, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        imgs.append(x)

    return np.vstack(imgs)

def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def train():
    train_data_dir = 'data/train'
    valid_data_dir = 'data/valid'
    nb_train_samples = 64
    nb_valid_samples = 32
    batch_size = 4
    epochs = 3

    img_width, img_height = 128, 128
    if backend.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

    model = create_model(input_shape)
    model.fit_generator(
        train_generator,
        steps_per_epoch=(nb_train_samples // batch_size),
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=(nb_valid_samples // batch_size)
    )

    scores = model.evaluate(valid_generator, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    json.dump(model.to_json(), open("model.json", "w"))
    model.save_weights('model.h5')

    return model

def test():
    images = [
        'data/train/trees/1.png',
        'data/train/trees/2.png',
        'data/train/trees/3.png',
        'data/train/trees/4.png',
        'data/train/trees/5.png',
        'data/train/trees/6.png',
        'data/train/nontrees/1.png',
        'data/train/nontrees/2.png',
        'data/train/nontrees/3.png',
        'data/train/nontrees/4.png',
        'data/train/nontrees/5.png',
        'data/train/nontrees/6.png',
    ]

    model = model_from_json(json.load(open('model.json', 'r')))
    model.load_weights('model.h5')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    images = load_images(images)
    classes = model.predict_classes(images, batch_size=12)

    print(classes)


img_width, img_height = 128, 128

if __name__ == '__main__':
    #train()
    test()