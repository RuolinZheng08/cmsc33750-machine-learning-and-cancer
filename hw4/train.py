#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

def main():
    if len(sys.argv) != 2:
        print('train.py [p1|p2|p4]')
        exit(1)

    model_name = sys.argv[1]
    print('model_name:', model_name)

    # load data differently depending on the part of the HW
    print('Load data')
    X_train, Y_train, X_test, Y_test = None, None, None, None
    if model_name == 'p1':
        X_train, Y_train, X_test, Y_test = load_data_tumor_only()
    elif model_name == 'p2':
        X_train, Y_train, X_test, Y_test = load_data_concat_normal()
    elif model_name == 'p4':
        X_train, Y_train, X_test, Y_test = load_data_tc_labels()

    print('X_train {}, Y_train {}, X_test {}, Y_test {}'.format(
        X_train.shape, Y_train.shape, X_test.shape, Y_train.shape
    ))

    # build model
    model = construct_model(num_features=X_train.shape[1],
    num_classes=Y_train.shape[1])

    # train
    train(model, X_train, Y_train, X_test, Y_test, model_name=model_name)

################################################################################

def load_data_tc_labels(): # for p4
    nt_train = np.load('data/nt_train.npy')
    nt_train_tc_labels = np.load('data/nt_train_tc_labels.npy')

    tc_train = np.load('data/tc_train.npy')
    tc_train_labels = np.load('data/tc_train_labels.npy')

    nt_test = np.load('data/nt_test.npy')
    nt_test_tc_labels = np.load('data/nt_test_tc_labels.npy')

    tc_test = np.load('data/tc_test.npy')
    tc_test_labels = np.load('data/tc_test_labels.npy')

    # concat data
    X_train = np.concatenate((nt_train, tc_train, nt_test), axis=0)
    Y_train = np.concatenate((nt_train_tc_labels, tc_train_labels, nt_test_tc_labels), axis=0)

    X_test = tc_test
    Y_test = tc_test_labels

    return X_train, Y_train, X_test, Y_test

def load_data_tumor_only(): # for p1
    nt_train = np.load('data/nt_train.npy')
    nt_train_tc_labels = np.load('data/nt_train_tc_labels.npy')

    tc_train = np.load('data/tc_train.npy')
    tc_train_labels = np.load('data/tc_train_labels.npy')

    nt_test = np.load('data/nt_test.npy')
    nt_test_tc_labels = np.load('data/nt_test_tc_labels.npy')

    tc_test = np.load('data/tc_test.npy')
    tc_test_labels = np.load('data/tc_test_labels.npy')

    # pull out tumor data
    nt_train_labels = np.load('data/nt_train_labels.npy')
    nt_test_labels = np.load('data/nt_test_labels.npy')

    train_tumor_indices = (nt_train_labels[:, 1] == 1)
    test_tumor_indices = (nt_test_labels[:, 1] == 1)

    nt_train = nt_train[train_tumor_indices]
    nt_test = nt_test[test_tumor_indices]

    nt_train_tc_labels = nt_train_tc_labels[train_tumor_indices]
    nt_test_tc_labels = nt_test_tc_labels[test_tumor_indices]

    # concat data
    X_train = np.concatenate((nt_train, tc_train, nt_test), axis=0)
    Y_train = np.concatenate((nt_train_tc_labels, tc_train_labels, nt_test_tc_labels), axis=0)

    X_test = tc_test
    Y_test = tc_test_labels

    return X_train, Y_train, X_test, Y_test

def load_data_concat_normal(): # for p2
    nt_train = np.load('data/nt_train.npy')

    tc_train = np.load('data/tc_train.npy')
    tc_train_labels = np.load('data/tc_train_labels.npy')

    nt_test = np.load('data/nt_test.npy')

    tc_test = np.load('data/tc_test.npy')
    tc_test_labels = np.load('data/tc_test_labels.npy')

    # pull out normal data
    nt_train_labels = np.load('data/nt_train_labels.npy')
    nt_test_labels = np.load('data/nt_test_labels.npy')

    train_normal_indices = (nt_train_labels[:, 1] == 0)
    test_normal_indices = (nt_test_labels[:, 1] == 0)

    nt_train = nt_train[train_normal_indices]
    nt_test = nt_test[test_normal_indices]

    normal_class_num = tc_test_labels.shape[1]
    nt_train_tc_labels = np.full((nt_train.shape[0], 1), normal_class_num)
    nt_test_tc_labels = np.full((nt_test.shape[0], 1), normal_class_num)

    # re-categorize
    num_classes = normal_class_num + 1
    tc_train_labels = np_utils.to_categorical(
        np.argmax(tc_train_labels, axis=1), num_classes)
    tc_test_labels = np_utils.to_categorical(
        np.argmax(tc_test_labels, axis=1), num_classes)
    nt_train_tc_labels = np_utils.to_categorical(nt_train_tc_labels, num_classes)
    nt_test_tc_labels = np_utils.to_categorical(nt_test_tc_labels, num_classes)

    # concat data
    X_train = np.concatenate((nt_train, tc_train, nt_test), axis=0)
    Y_train = np.concatenate((nt_train_tc_labels, tc_train_labels, nt_test_tc_labels), axis=0)

    X_test = tc_test
    Y_test = tc_test_labels

    return X_train, Y_train, X_test, Y_test

def construct_model(num_features, num_classes, weights=None):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid',
    input_shape=(num_features, 1)))

    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(),
                metrics=['accuracy'])

    if weights:
        model.load_weights(weights)

    return model

def train(model, X_train, Y_train, X_test, Y_test, model_name):
    checkpointer = ModelCheckpoint(filepath=model_name+'.autosave.model.h5', verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(model_name+'.training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    model.fit(X_train, Y_train,
    batch_size=20,
    epochs=400,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [checkpointer, csv_logger, reduce_lr])

    # serialize weights to HDF5
    model.save_weights(model_name+'.model.h5')
    print('Saved model to disk')

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()
