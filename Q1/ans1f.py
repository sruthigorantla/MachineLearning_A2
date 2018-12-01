################################################################################
#                                                                              #
#                           Code for question 1(b)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ans1a as a1a
import ans1c as a1c
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import sys

def rbf_kernel(x1, x2):
    """
        rbf_kernel(ndarray, ndarray) -> float
        
        Computes the RBF kernel of the specified examples.
        
        x1: (1, num_features) Input vector
        x2: (1, num_features) Input vector
        
        Returns: value
            value: The kernel value computed by applying the RBF kernel
    """
    # Initialize the value
    val = 0.0
    
    ############################## YOUR CODE HERE ##############################
    
    # Compute the kernel value
    
    # raise NotImplementedError
    gamma = 0.7
    val = np.exp(-gamma*np.linalg.norm(x1 - x2) ** 2)
    
    ############################################################################
    
    return val

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def load_original_data():
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    with open("./../Q1/data/SpambaseFull/train.txt","r") as fp:
        for line in fp:
            line = line.split(",")
            train_X.append(line[:-1])
            train_Y.append(int(line[-1]))

    with open("./../Q1/data/SpambaseFull/test.txt","r") as fp:
        for line in fp:
            line = line.split(",")
            test_X.append(line[:-1])
            test_Y.append(int(line[-1]))

    train_X = np.asarray(train_X, dtype=np.float32)
    train_Y = np.asarray(train_Y)
    # train_Y = np.expand_dims(train_Y,axis=1)
    test_X = np.asarray(test_X, dtype=np.float32)
    test_Y = np.asarray(test_Y)
    # test_Y = np.expand_dims(test_Y,axis=1)
    # train_X = np.astype(float)
    # test_X = np.astype(float)

    return train_X, train_Y, test_X, test_Y

def load_data():
    
    train_fold = []
    test_fold = []
    for i in [1,2,3,4,5]:
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        with open("./../Q1/data/SpambaseFolds/Fold"+str(i)+"/cv-train.txt","r") as fp:
            for line in fp:
                line = line.split()
                train_X.append(line[:-1])
                train_Y.append(int(float(line[-1])))

        with open("./../Q1/data/SpambaseFolds/Fold"+str(i)+"/cv-test.txt","r") as fp:
            for line in fp:
                line = line.split()
                test_X.append(line[:-1])
                test_Y.append(int(float(line[-1])))

        train_X = np.asarray(train_X, dtype=np.float32)
        train_Y = np.asarray(train_Y)
        train_Y = np.expand_dims(train_Y,axis=1)
        test_X = np.asarray(test_X, dtype=np.float32)
        test_Y = np.asarray(test_Y)
        test_Y = np.expand_dims(test_Y,axis=1)
        train_fold.append([train_X, train_Y])
        test_fold.append([test_X, test_Y])
        # train_X = np.astype(float)
        # test_X = np.astype(float)

    return train_fold, test_fold

def linear_kernel(x1, x2):
    """
        linear_kernel(ndarray, ndarray) -> float
        
        Computes the linear kernel of the specified examples.
        
        x1: (1, num_features) Input vector
        x2: (1, num_features) Input vector
        
        Returns: value
            value: The kernel value computed by applying the linear kernel
    """
    return np.matmul(x1, x2.T)


 
    
def show_decision_boundary(X, Y, X_train, Y_train, alphas, b, \
                           kernel=linear_kernel):
    """
        show_decision_boundary(ndarray, ndarray, ndarray, ndarray, ndarray, \
                                        float, function) -> None
    
        Shows decision boundary by plotting regions of positive and negative
        classes
        
        X: (num_examples, num_features) Feature matrix
        Y: (num_examples, 1) Label matrix
        X_train: (num_train_examples, 1) Feature matrix used for training
        Y_train: (num_train_examples, 1) Labels from training set
        alphas: (num_train_examples, 1) alphas obtained by training SVM
        b: Bias term
        kernel: Kernel function to use (same as svm_train)
    """
    # Some useful variables
    n, d = X.shape
    
    # Obtain the minimum and maximum coordinates
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    
    # Plot the prediction map
    colors = ['rs', 'bs']
    # for x in np.arange(x_min, x_max, 0.2).tolist():
    # for y in np.arange(y_min, y_max, 0.2).tolist():
    
    for i in range(len(X)):
        label, _ = a1c.svm_predict(X_train, Y_train, X[i], \
                             alphas, b, kernel)
        plt.plot(X[i], colors[int((label + 1) / 2)])

    plt.show()


    
    
if __name__ == '__main__':
    # Use ans1a to generate synthetic data as mentioned in question 1(b)
    train_fold, test_fold = load_data()
    # shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
    # train_X = train_X[shuffle_indices]
    # train_Y = train_Y[shuffle_indices]

    # Use svm_train to train on train_data using linear_kernel
    C_acc = []
    C_lst = [0.01, 0.1, 1, 10, 100]
    for C in C_lst:
        accuracy_avg = 0
        acc_avg = 0
        print("C: ",C)
        for ind, fold in enumerate(train_fold):
            X = fold[0]
            Y = fold[1]
            Y1 = []
            Y2 = []
            X1 = []
            X2 = []
            # Split into train_data [80%] and test_data [20%]
            for i in range(len(X)):
                if(Y[i] == 1):
                    Y1.append(1)
                    X1.append(X[i])
                else:
                    Y2.append(-1)
                    X2.append(X[i])
            # Obtaining the label y based on f
            Y1 = np.asarray(Y1)
            Y2 = np.asarray(Y2)
            X1 = np.asarray(X1)
            X2 = np.asarray(X2)
            train_X, train_Y = a1c.split_train(X1, Y1, X2, Y2)
            test_X, test_Y = a1c.split_test(X1, Y1, X2, Y2)
            # shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
            # train_X = train_X[shuffle_indices]
            # train_Y = train_Y[shuffle_indices]
            print(len(train_X))
            (alphas, b) = a1c.svm_train(train_X, train_Y, C, kernel=rbf_kernel)
            # Use svm_predict to obtain accuracy on train_data and test_data
            accuracy_train = 0
            accuracy_test = 0
            for i in range(len(train_X)):
                y,_ = a1c.svm_predict(train_X, train_Y, train_X[i], alphas, b, kernel=rbf_kernel)
                if(y == train_Y[i]):
                    accuracy_train += 1

            print("accuracy of training data: ",accuracy_train/len(train_X))
            for i in range(len(test_X)):
                y,_ = a1c.svm_predict(train_X, train_Y, test_X[i], alphas, b, kernel=rbf_kernel)
                if(y == test_Y[i]):
                    accuracy_test += 1
            print("accuracy of testing data: ",accuracy_test/len(test_X))
            accuracy_avg += accuracy_test/len(test_Y)
            acc_avg += accuracy_train/len(train_X)
        print("Total test accuracy_avg: ",accuracy_avg/5)
        print("Total train accuracy_avg: ",acc_avg/5)
        C_acc.append(accuracy_avg/5.0)
    C_best = C_lst[np.argmax(np.asarray(C_acc))]
    print("Best C is: ", C_best)
    
    train_X, train_Y, test_X, test_Y = load_original_data()
    count1 = 0
    count2 = 0
    for i in range(len(test_Y)):
        if(test_Y[i] == 1):
            count1 += 1
        else:
            count2 += 1
            # print(test_Y[i][0])
    print(count1, count2)
    # shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
    # train_X = train_X[shuffle_indices]
    # train_Y = train_Y[shuffle_indices]
    (alphas, b) = a1c.svm_train(train_X, train_Y, C_best, kernel=rbf_kernel)
    
    accuracy_train = 0.0
    for i in range(len(train_X)):
        y,_ = a1c.svm_predict(train_X, train_Y, train_X[i], alphas, b, kernel=rbf_kernel)
        if(y == train_Y[i]):
            accuracy_train += 1
    accuracy_test = 0.0
    print("accuracy of training data: ",float(accuracy_train)/len(train_Y))
    
    for i in range(len(test_X)):
        y,_ = a1c.svm_predict(train_X, train_Y, test_X[i], alphas, b, kernel=rbf_kernel)
        if(y == test_Y[i]):
            accuracy_test += 1
    print("accuracy of testing data: ",accuracy_test/len(test_Y))
    
    # Show the decision region using show_decision_boundary
    # show_decision_boundary(test_X, test_Y, train_X, train_Y, alphas, b)