#################################################################################
#                                                                               #
#                           Code for Question 2(a)                              #
#                                                                               #
#################################################################################

import numpy as np


def sigmoid(Z):
    """
        sigmoid(ndarray) -> ndarray
        
        Applies sigmoid to each entry of the matrix Z
        
        Z: (num_rows, num_cols) input matrix
        
        Returns: Z_hat
            Z_hat: (num_rows, num_cols) Matrix obtained by applying sigmoid to
                   each entry of Z 
    """
    # Initialize the output
    Z_hat = np.zeros(Z.shape)
    
    ########################### YOUR CODE HERE ################################
    
    # Compute Z_hat, avoid using a loop, otherwise it will be very slow
    
    # raise NotImplementedError
    Z_hat = 1/(1+np.exp(-Z))
    ###########################################################################
    
    return Z_hat
    
    
    
def sigmoid_grad(Z):
    """
        sigmoid_grad(ndarray) -> ndarray
        
        Let Z = sigmoid(X), be matrix obtained by applying sigmoid to another
        matrix X. This function computes sigmoid'(X).
        
        Z: (num_rows, num_cols) Sigmoid output
        
        Returns:
            Z_grad: (num_rows, num_cols) Computed gradient
    """
    # Initialize the output
    Z_grad = np.zeros(Z.shape)
    
    ########################### YOUR CODE HERE ################################
    
    # Compute Z_grad, avoid using a loop, otherwise it will be very slow
    
    # raise NotImplementedError
    Z_grad = sigmoid(Z)*(1-sigmoid(Z))
    ###########################################################################
    
    return Z_grad   
    


class Linear:
    """
        Class that implements a single linear layer
    """
    
    def __init__(self, num_inputs, num_outputs, act=sigmoid, \
                                                        act_grad=sigmoid_grad):
        """
            __init__(self, int, int, function, function) -> None
            
            num_inputs: Number of features in the input (excluding bias)
            num_outputs: Number of output neurons (excluding bias)
            act: Activation function to use
            act_grad: Function that computes gradient of the activation function
        """ 
        # Initialze variables that will be used later
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.act = act
        self.act_grad = act_grad
        
        ########################### YOUR CODE HERE #############################
    
        # Initialize the weight matrix W and bias vector b appropriately
        # self.W = ... and self.b = ...        

        # raise NotImplementedError
        self.W = np.random.rand(self.num_inputs,self.num_outputs)
        self.b = np.random.rand(1,self.num_outputs)

        ########################################################################
    
    
    def forward(self, X):
        """
            forward(Linear, ndarray) -> ndarray
            
            Computes the forward pass on this layer
            
            X: (num_examples, num_inputs) Input matrix for this layer. Each row
               corresponds to an example
            
            Returns: out
                out: (num_examples, num_outputs) Computed output activations
        """
        # Some useful variables
        num_examples = X.shape[0]
        
        # Initialze self.out, self is needed because it is used by backward
        self.out = np.zeros((num_examples, self.num_outputs))
        self.input = X # Will be used during backpropagation
        
        ########################### YOUR CODE HERE #############################
    
        # Compute the pre-activation outputs pre_acts  
        pre_acts = np.add(np.matmul(self.input,self.W),self.b)
        
        # Apply activations to pre_acts using self.act(pre_acts)      
        self.out = self.act(pre_acts)

        # raise NotImplementedError
        ########################################################################
        
        return self.out
        

