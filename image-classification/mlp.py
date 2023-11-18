import numpy as np
import matplotlib.pyplot as plt

from read_cifar import read_cifar, split_dataset

# Q10
def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    d_out = 2

    # Forward pass
    a0 = data # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))

    # Compute partial derivatives
    dC_dA2 = (2/d_out)*(a2-targets)
    dC_dZ2 = dC_dA2*a2*(1-a2)
    dC_dW2 = np.matmul(a1.T, dC_dZ2)/(a1.shape[0])
    dC_dB2 = dC_dZ2
    dC_dA1 = np.matmul(dC_dZ2, w2.T)
    dC_dZ1 = dC_dA1*a1*(1-a1)
    dC_dW1 = np.matmul(a0.T, dC_dZ1)/a0.shape[0]
    dC_dB1 = dC_dZ1

    w1_updated = w1 - learning_rate*dC_dW1 
    b1_updated = b1 - learning_rate*dC_dB1 
    w2_updated = w2 - learning_rate*dC_dW2
    b2_updated = b2 - learning_rate*dC_dB2

    return w1_updated, b1_updated, w2_updated, b2_updated, loss

# Q11
def one_hot(labels):
    max_label = max(labels)
    N_labels = len(labels)
    one_hot_matrix = np.zeros((N_labels, max_label+1))
    for i, l in enumerate(labels):
        one_hot_matrix[i,l] = 1
    return one_hot_matrix

# Q12
def learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate):
    """ 
    I added accuracy computation and returned it, because it is used in the next question (#Q13)
    """
    # Forward pass
    a0 = data_train # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = (e:=np.exp(z2))/(np.sum(e)) # output of the output layer (softmax activation function)
    predictions = a2  # the predicted values are the outputs of the output layer
    accuracy = sum(labels_train==np.argmax(predictions, axis=1))/len(labels_train)

    # Compute one-hot encoding of train labels
    one_hot_labels = one_hot(labels_train)

    # Compute loss (cross-entropy)
    loss = -np.sum(one_hot_labels*np.log(predictions))
    # print(loss)

    # Compute partial derivatives
    dC_dA2 = a2-one_hot_labels
    dC_dZ2 = dC_dA2*a2*(1-a2)
    dC_dW2 = np.matmul(a1.T, dC_dZ2)#/a1.shape[0]
    dC_dB2 = dC_dZ2
    dC_dA1 = np.matmul(dC_dZ2, w2.T)
    dC_dZ1 = dC_dA1*a1*(1-a1)
    dC_dW1 = np.matmul(a0.T, dC_dZ1)#/a0.shape[0]
    dC_dB1 = dC_dZ1

    w1_updated = w1 - learning_rate*dC_dW1 
    b1_updated = b1 - learning_rate*dC_dB1 
    w2_updated = w2 - learning_rate*dC_dW2
    b2_updated = b2 - learning_rate*dC_dB2

    return w1_updated, b1_updated, w2_updated, b2_updated, loss, accuracy

# Q13
def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = []
    train_losses = []
    for _ in range(num_epoch):
        w1, b1, w2, b2, running_loss, running_accuracy = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        train_accuracies.append(running_accuracy)
        train_losses.append(running_loss)
    return w1, b1, w2, b2, train_accuracies, train_losses

# Q14
def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    a0 = data_test 
    z1 = np.matmul(a0, w1) + b1 
    a1 = 1 / (1 + np.exp(-z1))  
    z2 = np.matmul(a1, w2) + b2  
    a2 = (e:=np.exp(z2))/(np.sum(e)) 
    predictions = a2  

    return sum(labels_test==np.argmax(predictions, axis=1))/len(labels_test)

# Q15
def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    d_in = data_train.shape[1]
    d_out = 10 #= number of classes

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    _, _, _, _, train_accuracies, train_losses = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    return train_accuracies, test_accuracy, train_losses

# Q16
def plot_learning_accuracy(path_batch='/home/vl/Documents/4A/liming_chen_deep/BE1/image-classification/data/cifar-10-python/cifar-10-batches-py'):
    num_epoch = 100
    
    m, l = read_cifar(path_batch)
    train_m, train_l, test_m, test_l = split_dataset(m,l,split=0.9)
    train_accuracies, _, _ = run_mlp_training(train_m, train_l, test_m, test_l, d_h=64, learning_rate=.1, num_epoch=num_epoch)
    
    plt.plot([i for i in range(1,num_epoch+1)], train_accuracies, marker='o', linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.title("Train accuracies of the MLP classifier accross epochs")
    plt.savefig('results/mlp.png')


if __name__ == '__main__':
    """ /!\ Add the batch path as parameter /!\ 
    (default value is path of my local configuration) """
    plot_learning_accuracy()