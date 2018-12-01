################################################################################
#                                                                              #
#                           Code for question 1(d)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ans1a as a1a


def complex_fn(x):
    """
        complex_fn(ndarray) -> {0, 1}
        
        Complex function that assigns class to x as explained in question 1(d)
        
        Returns: y
            y: Class label {+1, -1}
    """
    # Initialze the output
    y = 0
    
    ############################## YOUR CODE HERE ##############################
    
    # raise NotImplementedError
    for i in range(len(x)):
        if(np.linalg.norm(x) < 3):
            y = 1
        else:
            y = -1
    
    ############################################################################
    
    return y



if __name__ == '__main__':    
    # Use data_gen to generate synthetic data as specified in question 1(a)
    X, Y = a1a.data_gen(n=1000, f=complex_fn)

    # Plot the generated data using matplotlib
    
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y, s=5)
    # plt.savefig("./plots/q1d")
    plt.show()