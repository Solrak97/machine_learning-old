import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle


def split_to_tensors(num_vector, shape):
    tensors = []
    for entry in num_vector:
        entry = np.reshape(entry, shape)
        entry = torch.from_numpy(entry)
        entry = entry.type(torch.float32)
        tensors.append(entry)
    return tensors


def encode(array):
    encoded = np.zeros((array.size, array.max()+1))
    encoded[np.arange(array.size), array] = 1
    return encoded


def transform(x_train, x_test, y_train, y_test):
    x_train = torch.from_numpy(x_train.reshape((50000, 3, 32, 32))).type(torch.float32)
    x_test = torch.from_numpy(x_test.reshape((10000, 3, 32, 32))).type(torch.float32)
    
    y_train = torch.from_numpy(encode(np.array(y_train)))
    y_test = torch.from_numpy(encode(np.array(y_test)))

    return x_train, x_test, y_train, y_test



def load_data(DIRECTORY):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    names = [n.decode('utf-8')
             for n in unpickle(DIRECTORY+"/batches.meta")[b'label_names']]
    x_train = None
    y_train = []
    for i in range(1, 6):
        data = unpickle(DIRECTORY+"/data_batch_"+str(i))
        if i > 1:
            x_train = np.append(x_train, data[b'data'], axis=0)
        else:
            x_train = data[b'data']
        y_train += data[b'labels']
    data = unpickle(DIRECTORY+"/test_batch")
    x_test = data[b'data']
    y_test = data[b'labels']
    return names, x_train, y_train, x_test, y_test


def plot_tensor(tensor, perm=None):
    if perm == None:
        perm = (1, 2, 0)
    plt.figure()
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()
