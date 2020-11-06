import pandas as pd
import numpy as np

from keras.utils import np_utils

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


PL = 60484   # 1 + 60483 these are the width of the RNAseq datasets
P     = 60483   # 60483

NT_DIR = 'normal-tumor/'
TC_DIR = 'type-class/'

NT_TRAIN = 'nt_train2.csv'
NT_TEST = 'nt_test2.csv'

TC_TRAIN = 'type_18_300_train.csv'
TC_TEST = 'type_18_300_test.csv'

MODEL = 'tc1.autosave.model.h5'

def load_data(train_path, test_path, num_classes):
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,num_classes)
    Y_test = np_utils.to_categorical(df_y_test,num_classes)

    df_x_train = df_train[:, 1:PL].astype(np.float32)
    df_x_test = df_test[:, 1:PL].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test

NT_X_train, NT_Y_train, NT_X_test, NT_Y_test = load_data(
    NT_DIR + NT_TRAIN, NT_DIR + NT_TEST, num_classes=2
)

TC_X_train, TC_Y_train, TC_X_test, TC_Y_test = load_data(
    TC_DIR + TC_TRAIN, TC_DIR + TC_TEST, num_classes=36
)

DATA_DIR = 'data/'

np.save(DATA_DIR + 'nt_train', NT_X_train)
np.save(DATA_DIR + 'nt_train_labels', NT_Y_train)
np.save(DATA_DIR + 'nt_test', NT_X_test)
np.save(DATA_DIR + 'nt_test_labels', NT_Y_test)

np.save(DATA_DIR + 'tc_train', TC_X_train)
np.save(DATA_DIR + 'tc_train_labels', TC_Y_train)
np.save(DATA_DIR + 'tc_test', TC_X_test)
np.save(DATA_DIR + 'tc_test_labels', TC_Y_test)

