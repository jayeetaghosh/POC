# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.utils import get_file
from keras.optimizers import SGD
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import log_loss


def lstm_model(sequence_length, nb_features, nb_out=None):
    """

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()

    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=nb_out, activation='sigmoid'))

    return model