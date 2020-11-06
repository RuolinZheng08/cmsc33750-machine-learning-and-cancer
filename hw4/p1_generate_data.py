import numpy as np
from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score

MODEL = 'type-class/tc1.autosave.model.h5'
DATA_DIR = 'data/'
NT_TRAIN = 'nt_train.npy'
NT_TEST = 'nt_test.npy'
NT_TRAIN_LABELS = 'nt_train_labels.npy'
NT_TEST_LABELS = 'nt_test_labels.npy'
TC_TRAIN = 'tc_train.npy'
TC_TEST = 'tc_test.npy'
TC_TRAIN_LABELS = 'tc_train_labels.npy'
TC_TEST_LABELS = 'tc_test_labels.npy'

EPOCH = 400
BATCH = 20
CLASSES = 36

PL = 60484   # 1 + 60483 these are the width of the RNAseq datasets
P     = 60483   # 60483
DR    = 0.1      # Dropout rate

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

model.load_weights(MODEL)

nt_train = np.load(DATA_DIR + NT_TRAIN)
nt_test = np.load(DATA_DIR + NT_TEST)
# nt_train_labels = np.load(DATA_DIR + NT_TRAIN_LABELS)
# nt_test_labels = np.load(DATA_DIR + NT_TEST_LABELS)

nt_train_tc_probs = model.predict(nt_train)
nt_test_tc_probs = model.predict(nt_test)

np.save(DATA_DIR + 'nt_train_tc_labels', nt_train_tc_labels)
np.save(DATA_DIR + 'nt_test_tc_labels', nt_test_tc_labels)
