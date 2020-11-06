from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
from keras.models import Sequential, Model

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

def train(model, X_train, Y_train, X_test, Y_test):

