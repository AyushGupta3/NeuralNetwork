import random

import numpy as np
import mnist
from NeuralNetworkLibrary.NeuralNetworkClass import NeuralNetwork
from matplotlib import pyplot as plt

load_names = ['weights_ih.npy', 'weights_ho.npy', 'biases_o.npy',
              'biases_h.npy']
save_names = ['weights_ih1.npy', 'weights_ho1.npy', 'biases_o1.npy',
              'biases_h1.npy']
train_images = mnist.train_images()
train_images = train_images.reshape(60000, 1, 784)
train_labels = mnist.train_labels()
# random.shuffle(train_images)
# random.shuffle(train_labels)
train_images = np.resize(train_images, (100, 1, 784))
train_labels = np.resize(train_labels, 100)
test_labels = mnist.test_labels()
test_image = mnist.test_images().reshape(10000, 1, 784)
test_image = (np.ndarray.tolist((np.asfarray(test_image) / 255 * 0.99) + 1))
train_images = (np.ndarray.tolist((np.asfarray(train_images) / 255 * 0.99) + 1))


def main():
    new_train_labels = []
    for item in train_labels:
        new_train_labels.append(convert_to_output_list(item))

    data = wrap_data_together(train_images, new_train_labels)
    # print(data[1])
    nn = NeuralNetwork(784, 35, 10)
    nn.load(load_names)
    nn.StochasticGradientDescent(data, 1000, 0.0456567)
    # print(nn.evaluate(train_images[0][0]))
    # print(nn.feedforward(train_images[0][0]))
    nn.save(save_names)
    accuracy()
    # img = np.array(train_images[0]).reshape(28, 28)

    # plt.imshow(img, cmap ='gray')
    # plt.show()
    # test()


def test():
    data = train_images[6][0]
    # print(data)
    nn = NeuralNetwork(784, 200, 10)
    nn.load(save_names)
    img = np.array(data).reshape(28, 28)
    print(nn.evaluate(data))
    print(nn.feedforward(data))
    plt.imshow(img, cmap='gray')
    plt.show()


def convert_to_output_list(num):
    return [0.01 if i != num else 0.99 for i in range(10)]


def wrap_data_together(train_data, train_labels):
    lst = [x + [y] for x, y in zip(train_data, train_labels)]
    return lst


def accuracy():
    nn = NeuralNetwork(784, 30, 10)
    nn.load(save_names)

    correct = 0
    for i in range(len(test_image)):
        if nn.evaluate(test_image[i][0]) == test_labels[i]:
            correct += 1

    print(correct)


main()
