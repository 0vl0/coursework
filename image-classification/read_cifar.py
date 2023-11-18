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
def split_dataset(data, labels, split):
    """
    Split data into train / test sets using numpy's choice function.
    """
    N = len(data)
    train_indices = np.random.choice(N, int(N*split), replace=False)
    set_train = set(train_indices)
    test_indices = np.array([i for i in range(N) if i not in set_train])
    return data[train_indices,:], labels[train_indices], data[test_indices,:], labels[test_indices]

if __name__ == '__main__':
    m, l = read_cifar('/home/vl/Documents/4A/liming_chen_deep/BE1/image-classification/data/cifar-10-python/cifar-10-batches-py')
    print(f'm.shape = {m.shape}, l.shape = {l.shape}, l.size = {l.size}')
    train_m, train_l, test_m, test_l = split_dataset(m,l,0.2)
    print(f'train_m.shape = {train_m.shape}, test_m.shape = {test_m.shape}')
