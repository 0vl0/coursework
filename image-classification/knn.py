import numpy as np

from read_cifar import read_cifar, split_dataset
import collections
import os
import matplotlib.pyplot as plt

# Q1
def distance_matrix(m1,m2):
    """
    Compute distance matrix between matrices m1 and m2, representing images.
    Only matrix operations are used.
    ********
    Inputs:
    m1: np array
    m2: np array
    """
    m1_squared = np.matmul(np.sum(m1*m1, axis=1, keepdims=True), np.ones((1,m2.shape[0])))
    m2_squared = np.matmul(np.sum(m2*m2, axis=1, keepdims=True), np.ones((1,m1.shape[0]))).T
    m1_dot_m2 = -2*np.matmul(m1, m2.T)
    return np.sqrt(m1_squared+m1_dot_m2+m2_squared)

# Q2 - à tester
def knn_predict(dist, labels_train, k):
    """ 
    Predict labels of the test set using the distance matrix between the train and test sets.
    *******
    Inputs:
    dist: distance matrix
    labels_train: labels of the training set
    k: parameter of the knn classifier
    """
    sorted_indices = np.argsort(dist, axis=0)
    return [collections.Counter(labels_train[sorted_indices[:k,j]]).most_common()[0][0] for j in range(len(dist[0]))]

# Q3 
def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    """ 
    Return the classification rate of the knn-classifier.
    """
    labels_predicted = knn_predict(distance_matrix(data_train, data_test), labels_train, k)
    print(f'k = {k}, accuracy = {sum(labels_predicted == labels_test) / len(labels_test)}')
    return sum(labels_predicted == labels_test) / len(labels_test)

# Q4
def plot_accuracy(path_batch='/home/vl/Documents/4A/liming_chen_deep/BE1/image-classification/data/cifar-10-python/cifar-10-batches-py', split_factor=0.9):
    """ 
    Plot the accuracy of the knn clasifier against k.
    """

    if not os.path.isdir('results'): os.mkdir('results')

    m, l = read_cifar(path_batch)
    train_m, train_l, test_m, test_l = split_dataset(m,l,split_factor)
    accuracies, indices = [evaluate_knn(train_m, train_l, test_m, test_l, k) for k in range(1,21)], [i for i in range(1,21)]
    
    plt.plot(indices, accuracies, marker='o', linestyle='-')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of the KNN classifier against k")
    plt.savefig('results/knn.png')


if __name__ == '__main__':
    plot_accuracy()