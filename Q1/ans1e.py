################################################################################
#                                                                              #
#                           Code for question 1(e)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ans1a as a1a
import ans1d as a1d
import ans1b as a1b


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
    
    
def split_train(X1, y1, X2, y2):
    ind = int(0.8*len(X1))
    X1_train = X1[:ind]
    y1_train = y1[:ind]
    X2_train = X2[:ind]
    y2_train = y2[:ind]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    ind = int(0.8*len(X1))
    X1_test = X1[ind:]
    y1_test = y1[ind:]
    X2_test = X2[ind:]
    y2_test = y2[ind:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test

if __name__ == '__main__':    
    # Generate 1000 synthetic data points as in 1(d)
    X, Y = a1a.data_gen(n=1000, f=a1d.complex_fn)
    Y1 = []
    Y2 = []
    X1 = []
    X2 = []
    # Split into train_data [80%] and test_data [20%]
    for i in range(len(X)):
        # X[i][0] = np.random.uniform(-lims[0],lims[0])
        # X[i][1] = np.random.uniform(-lims[1],lims[1])
        # Y[i] = f(X[i])
        if(Y[i] == 1):
            Y1.append(1)
            X1.append(X[i])
        else:
            Y2.append(-1)
            X2.append(X[i])
    # Obtaining the label y based on f
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
    
    plt.show()
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    
    
    # Use a1b.svm_train with rbf_kernel on 80% of data - train_data
    train_X, train_Y = split_train(X1, Y1, X2, Y2)
    test_X, test_Y = split_test(X1, Y1, X2, Y2)
    # Implement rbf_kernel function
    
    (alphas, b) = a1b.svm_train(train_X, train_Y, kernel=rbf_kernel)
    # Compute accuracy on train_data and test_data
    accuracy_train = 0
    accuracy_test = 0
    for i in range(len(train_X)):
        y, _ = a1b.svm_predict(train_X, train_Y, train_X[i], alphas, b, kernel=rbf_kernel)
        if(y == train_Y[i]):
            accuracy_train += 1
    print("accuracy of training data: ",float(accuracy_train)/len(train_X))
    for i in range(len(test_X)):
        y, _ = a1b.svm_predict(train_X, train_Y, test_X[i], alphas, b, kernel=rbf_kernel)
        if(y == test_Y[i]):
            accuracy_test += 1
    print("accuracy of testing data: ",float(accuracy_test)/len(test_X))
    # Show the decision region using show_decision_boundary
    a1b.show_decision_boundary(X, Y, train_X, train_Y, alphas, b, kernel=rbf_kernel)