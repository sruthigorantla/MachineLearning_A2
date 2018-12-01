#################################################################################
#                                                                               #
#                           Code for Question 2(b)                              #
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
    
    # Copy your implementation from Ans 2(a) here
    
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
    
    # Copy your implementation from Ans 2(a) here
    
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
    
        # Copy your implementation from Ans 2(a) here      

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
        
        # Initialze self.out, self is needed beacuse it is used by backward
        self.out = np.zeros((num_examples, self.num_outputs))
        self.input = X # Will be used during backpropagation
        
        ########################### YOUR CODE HERE #############################
    
        # Copy your implementation from ans2a here
        
        # Compute the pre-activation outputs pre_acts  
        pre_acts = np.add(np.matmul(self.input,self.W),self.b)
        
        # Apply activations to pre_acts using self.act(pre_acts)      
        self.out = self.act(pre_acts)
        # raise NotImplementedError
        ########################################################################
        
        return self.out


    def backward(self, delta_out):
        """
            backward(Linear, ndarray) -> ndarray
            Computes the gradient of the weights associated with this layer.
            Returns the error associated with input.

            delta_out: (num_examples, num_output) Error associated with the output units

            Returns: delta_in
                delta_in: (num_examples, num_inputs) Errors associated with the input
                          units
        """
        # Some useful variables
        num_examples = delta_out.shape[0]
        
        # Initialize the variables
        self.W_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)
        # print(self.b.shape)
        delta_in = np.zeros((num_examples, self.num_inputs))

        ########################### YOUR CODE HERE #############################
    
        # Compute self.W_grad, self.b_grad and delta_in

        # raise NotImplementedError
        Z = self.input
        delta_in = np.matmul(delta_out, self.W.T)*sigmoid_grad(Z)
        self.W_grad = np.matmul(self.input.T, delta_out)/num_examples
        self.b_grad = np.sum(delta_out,axis = 0)/num_examples
        # self.b_grad = delta_out
        self.b_grad = np.expand_dims(self.b_grad, axis=0)
        
        ########################################################################
        
        return delta_in


    def step(self, learning_rate=1e-2):
        """
            step(Linear, float) -> None

            Updates the weights of this layer using gradients computed by the
            backward funciton by applying a single step of gradient descent

            learning_rate: The learning rate used for gradient descent
        """
        ########################### YOUR CODE HERE #############################
    
        # Update self.W and self.b based on self.W_grad and self.b_grad
        self.W = self.W - learning_rate*self.W_grad
        self.b = self.b - learning_rate*self.b_grad
        
        # raise NotImplementedError
    
        ########################################################################
        

