from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from skimage import data
from skimage.color import rgb2gray
import numpy as np

def grayscale(data):
    data_holder = []
    for i in range(len(data)):
        data_holder.append(rgb2gray(data[i]))
    data_holder = np.array(data_holder)
    data_holder = data_holder.reshape(len(data_holder), 32, 32, 1)
    return data_holder


IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# Loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalization
x_train = x_train/255.0
x_test = x_test/255.0
y_train = to_categorical(y_train, CLASSES)
y_test = to_categorical(y_test, CLASSES)

# convert to grayscale
x_train = grayscale(x_train)
x_test = grayscale(x_test)

# Create the network:
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = x_train.shape[1:]))
#model.add(Conv2D(32, (3, 3), padding='same', input_shape = x_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(CLASSES))
model.add(Activation('softmax'))
model.summary()

# Train the model
model.compile(OPTIM,loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT,
          verbose=VERBOSE)

model.save('cifar80%accuracy.model')
score = model.evaluate(x_test, y_test)
