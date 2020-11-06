import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score

EPOCH = 400
BATCH = 20
CLASSES = 36

PL = 60484   # 1 + 60483 these are the width of the RNAseq datasets
P     = 60483   # 60483
DR    = 0.1      # Dropout rate

DATA_DIR = 'data/'
NT_TRAIN = 'nt_train.npy'
NT_TEST = 'nt_test.npy'
NT_TRAIN_TC_LABELS = 'nt_train_tc_labels.npy'
NT_TEST_TC_LABELS = 'nt_test_tc_labels.npy'
TC_TRAIN = 'tc_train.npy'
TC_TEST = 'tc_test.npy'
TC_TRAIN_LABELS = 'tc_train_labels.npy'
TC_TEST_LABELS = 'tc_test_labels.npy'

# load data
nt_train = np.load(DATA_DIR + NT_TRAIN)
tc_train = np.load(DATA_DIR + TC_TRAIN)
nt_test = np.load(DATA_DIR + NT_TEST) # nt_test is generated by a TC model
nt_train_tc_labels = np.load(DATA_DIR + NT_TRAIN_TC_LABELS)
tc_train_labels = np.load(DATA_DIR + TC_TRAIN_LABELS)
nt_test_tc_labels = np.load(DATA_DIR + NT_TEST_TC_LABELS)

tc_test = np.load(DATA_DIR + TC_TEST)
tc_test_labels = np.load(DATA_DIR + TC_TEST_LABELS)

X_train = np.concatenate((nt_train, tc_train, nt_test), axis=0)
Y_train = np.concatenate((nt_train_tc_labels, tc_train_labels, nt_test_tc_labels), axis=0)

X_test = tc_test
Y_test = tc_test_labels

# define model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
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
model.add(Dense(CLASSES))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# set up a bunch of callbacks to do work during model training..

checkpointer = ModelCheckpoint(filepath='tc2.autosave.model.h5', verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger('tc2.training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit(X_train, Y_train,
                    batch_size=BATCH,
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks = [checkpointer, csv_logger, reduce_lr])

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# serialize weights to HDF5
model.save_weights("tc2.model.h5")
print("Saved model to disk")
