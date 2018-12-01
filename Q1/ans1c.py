################################################################################
#                                                                              #
#                           Code for question 1(b)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ans1a as a1a
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import sys


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



def svm_train(X, Y, C, kernel=linear_kernel):
    """
        svm_train(ndarray, ndarray, function) -> ndarray, float
        
        Trains the hard SVM on provided data.
        
        X: (num_examples, num_features) Input feature matrix
        Y: (num_examples, 1) Labels
        kernel: Function that computes kernel(x1, x2)
        
        Returns: (alphas, b)
            alphas: (num_examples, 1) alphas obtained by solving SVM
            b: bias term for SVM
    """
    # Some useful variables
    n, d = X.shape
    
    # Initialize the parameters
    alphas = np.random.random(size=(n, 1))
    b = 0
    
    ############################## YOUR CODE HERE ##############################
    
    # Prepare your optimization problem here. 
    # Install cvxopt by using:
    #   pip install cvxopt
    # Use cvxopt (http://cvxopt.org/examples/tutorial/qp.html) to solve it.
    # Use the solution to obtain alphas and b
    
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(X[i], X[j])

    P = matrix(np.outer(Y,Y)*K,tc='d')
    q = matrix(np.ones(n)*-1,(n,1),tc='d')
    A = matrix(Y, (1,n),tc='d')
    b = matrix(0.0,(1,1),tc='d')

    if C is 0:
        G = matrix(np.diag(np.ones(n) * -1))
        h = matrix(np.zeros(n))
    else:
        tmp1 = np.diag(np.ones(n) * -1)
        tmp2 = np.identity(n)
        G = matrix(np.vstack((tmp1, tmp2)), tc='d')
        tmp1 = np.zeros(n)
        tmp2 = np.ones(n) * C
        h = matrix(np.hstack((tmp1, tmp2)), tc='d')

    # print(np.linalg.matrix_rank(P))
    # print(np.linalg.matrix_rank(q))
    # print(np.linalg.matrix_rank(A))
    # print(np.linalg.matrix_rank(b))
    # print(np.linalg.matrix_rank(G))
    # print(np.linalg.matrix_rank(h))

    solution = solvers.qp(P, q, G, h,  A, b)
    alphas = np.ravel(solution['x'])

    new_alphas = []
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    for i in range(len(sv)):
        if(sv[i]):
            new_alphas.append(alphas[i])
    sv_y = Y[sv]
    sv_x = X[sv]
    b = 0.0
    for i in range(len(new_alphas)):
        b += sv_y[i]
        b -= np.sum(new_alphas * sv_y * K[ind[i],sv])
    
    b /= len(new_alphas)

    ############################################################################
    
    return (alphas, b)
    



def svm_predict(X_train, Y_train, X, alphas, b, kernel=linear_kernel):
    """
        svm_predict(ndarray, ndarray, ndarray, ndarray, float, function) -> \
                                                         float, float
        
        Predicts the output labels Y based on input X and trained SVM
        
        X_train: (num_examples, num_features) Training feature matrix
        Y_train: (num_examples, 1) Training labels
        X: (1, num_features) Features of input test example
        alphas: (num_examples, 1) alphas obtained by training SVM
        b: Bias term
        kernel: Kernel function to use (same as svm_train)
        
        Returns: (Y, fval)
            Y: Predicted label {+1, -1} for X
            fval: The value of function which was thresholded to get Y
    """
    # Some useful variables
    n, d = X_train.shape
    
    # Intialize output
    fval = 0    # The value equivalent to wx + b
    Y = 0   # Label obtained by thresholding fval
    
    ############################## YOUR CODE HERE ##############################
    
    # Compute the output prediction Y
    
    # raise NotImplementedError
    for i in range(len(alphas)):
        try:
            fval += alphas[i] * Y_train[i] * kernel(X_train[i], X)
        except IndexError:
            print(len(alphas), len(Y_train), len(X_train), len(X))
            sys.exit(0)
    fval += b
    
    if(fval <= 0):
        Y = -1
    elif(fval > 0):
        Y = 1
    
    ############################################################################
    
    return (Y, fval)
    
    
    
    
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
    for x in np.arange(x_min, x_max, 0.2).tolist():
        for y in np.arange(y_min, y_max, 0.2).tolist():
            label, _ = svm_predict(X_train, Y_train, np.asarray([[x, y]]), \
                                 alphas, b, kernel)
            plt.plot(x, y, colors[int((label + 1) / 2)])
    
    plt.show()

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
    # Use ans1a to generate synthetic data as mentioned in question 1(b)
    X, Y = a1a.data_gen(n=1000)
    # with open("data.pickle","rb") as fp:
    #     (X,Y) = pickle.load(fp)
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
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
    
    plt.show()
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    train_X, train_Y = split_train(X1, Y1, X2, Y2)
    test_X, test_Y = split_test(X1, Y1, X2, Y2)
    # Use svm_train to train on train_data using linear_kernel
    for C in [0.1, 1, 10]:
        for p in [0, 0.1, 0.3]:
            print("C: ",C)
            print("p: ",p)
            new_train_Y = []
            count = 0
            for i in range(len(train_Y)):
                if(np.random.rand(1) <= p):
                    if(train_Y[i] == 1):
                        count += 1
                        new_train_Y.append(-1)
                    else:
                        new_train_Y.append(1)
                else:
                    new_train_Y.append(train_Y[i])
            print(count)
            plt.scatter(train_X[:, 0], train_X[:, 1], marker='.', c=new_train_Y, s=5)
    
            # plt.show()
            new_train_Y = np.asarray(new_train_Y)   
            (alphas, b) = svm_train(train_X, new_train_Y, C)
            # show_decision_boundary(train_X, new_train_Y, train_X, new_train_Y, alphas, b)
            # Use svm_predict to obtain accuracy on train_data and test_data
            accuracy_train = 0
            accuracy_test = 0
            for i in range(len(train_X)):
                y,_ = svm_predict(train_X, new_train_Y, train_X[i], alphas, b)
                if(y == train_Y[i]):
                    accuracy_train += 1
            print("accuracy of training data: ",accuracy_train/len(train_X))
            for i in range(len(test_X)):
                y,_ = svm_predict(train_X, new_train_Y, test_X[i], alphas, b)
                if(y == test_Y[i]):
                    accuracy_test += 1
            print("accuracy of testing data: ",accuracy_test/len(test_X))
    # Show the decision region using show_decision_boundary
    # show_decision_boundary(train_X, train_Y, train_X, train_Y, alphas, b)