from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from skimage import data
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf

classification_list = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def grayscale(data):
    data_holder = []
    for i in range(len(data)):
        data_holder.append(rgb2gray(data[i]))
    data_holder = np.array(data_holder)
    data_holder = data_holder.reshape(len(data_holder), 32, 32, 1)
    return data_holder


# Loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_test = to_categorical(y_test, 10)

# convert to grayscale
x_train = grayscale(x_train)
x_test_modif = grayscale(x_test)

model = tf.keras.models.load_model('cifar80%accuracy.model')
prediction = model.predict(x_test_modif)

plt.figure(figsize=(15,10))
for i in range(10):
    random = np.random.randint(0,10000)
    plt.subplot(5,2,i+1)
    plt.imshow(x_test[random])
    index = np.argmax(prediction[random])
    plt.ylabel('{}'.format(classification_list[index]))
plt.savefig('cifar_prediction.png')
