# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.utils import get_file
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

from load_cifar10 import load_cifar10_data

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras

    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Download ImageNet weights file
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    weights_path = get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        WEIGHTS_PATH,
        cache_subdir='/home/jghosh/keras-finetune/imagenet_models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')

    # Loads ImageNet pre-trained data
    model.load_weights(weights_path)
    # model.load_weights('imagenet_models/vgg16_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001 # JG moved this to train.py
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# if __name__ == '__main__':
#
#     # Example to fine-tune on 3000 samples from Cifar10
#
#     img_rows, img_cols = 224, 224 # Resolution of inputs
#     channel = 3
#     num_classes = 10
#     batch_size = 16 #16
#     nb_epoch = 100
#
#     # Load Cifar10 data. Please implement your own load_data() module for your own dataset
#     X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
#
#     # Load our model
#     model = vgg16_model(img_rows, img_cols, channel, num_classes)
#
#     # Start Fine-tuning
#     model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               shuffle=True,
#               verbose=1,
#               validation_data=(X_valid, Y_valid),
#               )
#
#     # Make predictions
#     predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
#
#     # Cross-entropy loss score
#     score = log_loss(Y_valid, predictions_valid)
#     print("Log loss score for validation", score)
#     print("Done!")

