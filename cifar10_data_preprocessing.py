import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """
    load the cifar-10 data.
    """

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_labels, test_data, test_labels, label_names.
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b"label_names"]
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b"data"]
        else:
            cifar_train_data = np.vstack(
                (cifar_train_data, cifar_train_data_dict[b"data"])
            )
        cifar_train_labels += cifar_train_data_dict[b"labels"]

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b"data"]
    cifar_test_labels = cifar_test_data_dict[b"labels"]

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    return (
        cifar_train_data,
        cifar_train_labels,
        cifar_test_data,
        cifar_test_labels,
        cifar_label_names,
    )


def split_dataset(data, labels, num_dev=10000):
    """
    Shuffle train data and labels, then split into train_data and dev_data.
    """

    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    dev_data = data[:num_dev]
    dev_labels = labels[:num_dev]
    train_data = data[num_dev:]
    train_labels = labels[num_dev:]
    return dev_data, dev_labels, train_data, train_labels


def one_hot_encoding(labels, classes):
    """
    Args:
    1. labels : The label of each image is encoded to integer(0~9).
        (ex: 'airplane' : 0, 'automobile' : 1)
        Its shape is (number of data, 1).
    2. classes : Number of classes(10).

    Returns:
    1. one_hot_vector : Label's one-hot vector version.
        For example, label 'airplane(0)' is encoded to [1. 0. 0. ... 0.] through this function.
        Its shape is (number of data, 10).
    """

    one_hot_vector = np.zeros((labels.shape[0], classes.shape[0]))
    for i, label in enumerate(labels):
        one_hot_vector[i, label] = 1.0
    return one_hot_vector


def normalize_image(data):
    """
    Return normalized image using min-max normalization.
    The pixel values of input image are rescaled from 0~255 to 0~1.
    """

    data_min = data.min(axis=(1, 2), keepdims=True)
    data_max = data.max(axis=(1, 2), keepdims=True)
    norm_data = (data - data_min) / (data_max - data_min)
    return norm_data
