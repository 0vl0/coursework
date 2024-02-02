import pickle 
import numpy as np
import pathlib

# Q2
def read_cifar_batch(path_batch: str):
    """ 
    Read a batch file and return the batch of images and corresponding labels
    *******
    Inputs:
      path_batch: path of the batch
    """
    with open(path_batch, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    matrix = dict[b'data']
    labels = np.asarray(dict[b'labels'], dtype=np.int64)
    return matrix.astype(np.float32), labels

# Q3
def read_cifar(path_batches):
    """ 
    Read data from all batches and return it
    """
    matrix, labels = read_cifar_batch(path_batches+'/'+f'data_batch_1')
    batch_names = [f'data_batch_{i}' for i in range(2,6)] + ['test_batch']
    for b in batch_names:
        matrix, labels = np.append(matrix, (t := read_cifar_batch(path_batches+'/'+b))[0], axis=0), np.append(labels, t[1])
    return matrix, labels

# Q4
def split_dataset(data, labels, split_train, split_val):
    """
    Split data into train / val / test sets using numpy's choice function.
    *******
    Params:
    split_train: fraction of data used in train set
    split_val: fraction of data used in validation set

    """
    N = len(data)
    train_indices, val_indices, test_indices = (p:=np.random.permutation(np.arange(N)))[:int(N*split_train)], p[int(N*split_train):int(N*(split_train+split_val))], p[int(N*(split_train+split_val)):]
    return data[train_indices,:], labels[train_indices], data[val_indices,:], labels[val_indices], data[test_indices,:], labels[test_indices]

if __name__ == '__main__':
    m, l = read_cifar('/home/vl/Documents/4A/liming_chen_deep/BE1/image-classification/data/cifar-10-python/cifar-10-batches-py')
    print(f'm.shape = {m.shape}, l.shape = {l.shape}, l.size = {l.size}')
    train_m, train_l, val_m, val_l, test_m, test_l = split_dataset(m,l,0.8,0.1)
    print(f'train_m.shape = {train_m.shape}, val_m.shape = {val_m.shape}, test_m.shape = {test_m.shape}')
