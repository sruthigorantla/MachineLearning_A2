################################################################################
#                                                                              #
#                           Code for question 1(a)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt

def dummy_fn(x):
    """
        dummy_fn(ndarray) -> {0, 1}
        
        Simple function that assigns class to x as: y = sign(sum(x))
        
        Returns: y
            y: Class label {+1, -1}
    """
    if np.sum(x) == 0:
        return -1
    return np.sign(np.sum(x))
    



def data_gen(n=100, f=dummy_fn, lims=[5, 5]):
    """
        data_gen(int, function, list) -> (ndarray, ndarray)
        
        Generates synthetic data by using the classification rule specified by
        the function f. 
        
        n: Number of data points to generate
        f: Function that takes a d-dimensional vector x as input and produces a
           the class label for that point {+1, -1}
        lims: lims[i] is used to get upper and lower limit on domain for the ith
              dimension. The ith dimension of x will be uniformly sampled from 
              the interval [-lim[i], lim[i]] for all examples
        Length of dims (=d) should be taken as the dimension of input vectors.              
    
        Returns: (X, Y)
            X: (n, d) Feature matrix
            Y: (n, 1) Label matrix    
    """
    # Some useful variables
    d = len(lims)
    
    # Intialize the data
    X = np.zeros((n, d))
    Y = np.zeros((n, 1))
    
    ############################## YOUR CODE HERE ##############################
    
    # Generate each data point by:
    # Getting a vector x by sampling each component independently and
    # uniformly from the domain specified by lims
    for i in range(len(X)):
        X[i][0] = np.random.uniform(-lims[0],lims[0])
        X[i][1] = np.random.uniform(-lims[1],lims[1])
        Y[i] = f(X[i])
    # Obtaining the label y based on f
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
    plt.savefig("./plots/q1")
    # raise NotImplementedError
    
    ############################################################################
    
    return (X, Y)
    
    
    
if __name__ == '__main__':
    # Implement the data_gen function first
    
    # Now use data_gen to generate synthetic data as specified in question 1(a)
    X, Y = data_gen(n=1000)
    
    # Plot the generated data using matplotlib
    
    
    
    
    
