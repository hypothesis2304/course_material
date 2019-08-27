import os
import gzip
import numpy as np

def load_data():

    train_image_path = './data/train-images-idx3-ubyte'
    train_label_path = './data/train-labels-idx1-ubyte'
    test_image_path = './data/t10k-images-idx3-ubyte'
    test_label_path = './data/t10k-labels-idx1-ubyte'


    with gzip.open(train_label_path, 'rb') as tr_label:
        train_labels = np.frombuffer(tr_label.read(), dtype=np.uint8, offset=8)

    with gzip.open(train_image_path, 'rb') as tr_image:
        train_images = np.frombuffer(tr_image.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 784)

    with gzip.open(test_label_path, 'rb') as ts_label:
        test_labels = np.frombuffer(ts_label.read(), dtype=np.uint8, offset=8)

    with gzip.open(test_image_path, 'rb') as ts_image:
        test_images = np.frombuffer(ts_image.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 784)

    return train_images, train_labels, test_images, test_labels


