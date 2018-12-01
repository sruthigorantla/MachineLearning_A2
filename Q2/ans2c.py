#################################################################################
#                                                                               #
#                           Code for Question 2(c)                              #
#                                                                               #
#################################################################################

import numpy as np
import ans2b as a2b

class Net2c:
    """
        Class that defines the neural network for question 2(c)
    """

    def __init__(self):
        """
            __init__(Net2c) -> None
            
            Intializes the network.
        """
        ############################# YOUR CODE HERE ###########################

        # Define a single hidden layer neural network here.
        # Input layer contains 57 neurons + 1 bias since input has 57 features
        # Hidden layer 1 should contain 20 neurons + 1 bias, use sigmoid activation
        # Output layer should contain one neuron since it is a binary classifi-
        # -cation problem

        # Do something like
        #   self.input_layer = a2b.Linear(correct sizes and activations here)
        #   self.output_layer = a2b.Linear(correct sizes & activations here)
        # 
        # Do not change the name of these layers
        
        self.input_layer = a2b.Linear(57,20)
        self.output_layer = a2b.Linear(20,1)
        # raise NotImplementedError

        ########################################################################


    def forward(self, X):
        """
            forward(Net2c, ndarray) -> ndarray

            Computes the forward pass on the network to return the outputs

            X: (num_examples, num_features) Input feature matrix

            Returns:
                preds: (num_examples, 1) Predicted labels 
        """
        # Some useful variables
        num_examples = X.shape[0]
        
        # Initialize the predictions
        preds = np.zeros((num_examples, 1))

        ############################# YOUR CODE HERE ###########################

        # Run the forward pass by calling forward method of individual layers
        # in succession

        # raise NotImplementedError
        output1 = self.input_layer.forward(X)
        preds = self.output_layer.forward(output1)
        ########################################################################
        # print("Preds: ",preds.shape)
        return preds
        


    def backward(self, preds, Y):
        """
            backward(Net2c, ndarray, ndarray) -> float

            Runs the backward pass to compute gradients. Returns the cost
            
            preds: (num_examples, 1) Predictions made by neural network
            Y: (num_examples, 1) True labels

            Returns: cost
                cost: Cost incurred by the network
        """
        # Initialize the cost
        cost = 0.0

        ############################# YOUR CODE HERE ###########################

        # Compute the cost
        cost = 0.5*(np.linalg.norm(preds-Y)**2)

        # Compute delta term for output layer
        sigmoid_grad = preds*(1-preds)
        # print("sigmoid_grad: ",sigmoid_grad.shape)
        delta_out = -np.subtract(Y,preds)*sigmoid_grad
        # print("Delta out: ",delta_out.shape)

        # Run backward pass on on layers in the correct order
        delta_hidden = self.output_layer.backward(delta_out)
        # print("Delta hidden: ",delta_hidden.shape)

        delta_in = self.input_layer.backward(delta_hidden)
        # print("Delta in: ",delta_in.shape)
        # raise NotImplementedError
    
        ########################################################################

        return cost


    def step(self, learning_rate=1e-2):
        """
            step(Net2c, float) -> None

            Executes on step of gradient descent on the network

            learning_rate: Learning rate to be used by gradient descent
        """
        ############################# YOUR CODE HERE ###########################

        # Call the step method for each layer
        self.input_layer.step(learning_rate)
        self.output_layer.step(learning_rate)

        # raise NotImplementedError
    
        ########################################################################
        


##################### DO NOT MODIFY ANYTHING BELOW THIS LINE ###################    

if __name__ == '__main__':
    
    # Generate some data randomly
    X = np.random.random((10, 57))
    Y = np.random.choice([0, 1], 10).reshape((-1, 1))
    
    # Instantiate a network
    net = Net2c()
    
    # Compute the gradients
    preds = net.forward(X)
    cost = net.backward(preds, Y)
    input_W_grad = np.copy(net.input_layer.W_grad)
    input_b_grad = np.copy(net.input_layer.b_grad)
    output_W_grad = np.copy(net.output_layer.W_grad)
    output_b_grad = np.copy(net.output_layer.b_grad)
    
    # Check the first layer gradients for W
    eps = 1e-4
    for i in range(net.input_layer.W.shape[0]):
        for j in range(net.input_layer.W.shape[1]):
            # Compute positive deviation cost
            net.input_layer.W[i, j] += eps
            cost_pos = net.backward(net.forward(X), Y)
            
            # Compute the negative deviation cost
            net.input_layer.W[i, j] -= 2 * eps
            cost_neg = net.backward(net.forward(X), Y)
            
            # Compute the numerical gradient
            grad_calc = (cost_pos - cost_neg) / (2 * eps)
            
            # Check if it is within tolerance limit
            if np.abs(grad_calc - input_W_grad[i, j]) >= eps:
                print('Grad-Check Failed at (', i, ',', j, ') in input layer')
                print('Backpropagation:', input_W_grad[i, j], \
                      'Actual:', grad_calc)
                exit()
            
            # Restore original W value
            net.input_layer.W[i, j] += eps
            
    # Check first layer gradients for bias
    # print(net.input_layer.b.shape)
    for i in range(net.input_layer.b.shape[0]):
        # Compute positive deviation cost
        net.input_layer.b[i, 0] += eps
        cost_pos = net.backward(net.forward(X), Y)
        
        # Compute the negative deviation cost
        net.input_layer.b[i, 0] -= 2 * eps
        cost_neg = net.backward(net.forward(X), Y)
        
        # Compute the numerical gradient
        grad_calc = (cost_pos - cost_neg) / (2 * eps)
        
        # Check if it is within tolerance limit
        if np.abs(grad_calc - input_b_grad[i, 0]) >= eps:
            print('Grad-Check Failed at', i, 'in input layer')
            print('Backpropagation:', input_b_grad[i, 0], \
                  'Actual:', grad_calc)
            exit()
        
        # Restore original b value
        net.input_layer.b[i, 0] += eps
    
    # Check the output layer gradients for W
    eps = 1e-4
    for i in range(net.output_layer.W.shape[0]):
        for j in range(net.output_layer.W.shape[1]):
            # Compute positive deviation cost
            net.output_layer.W[i, j] += eps
            cost_pos = net.backward(net.forward(X), Y)
            
            # Compute the negative deviation cost
            net.output_layer.W[i, j] -= 2 * eps
            cost_neg = net.backward(net.forward(X), Y)
            
            # Compute the numerical gradient
            grad_calc = (cost_pos - cost_neg) / (2 * eps)
            
            # Check if it is within tolerance limit
            if np.abs(grad_calc - output_W_grad[i, j]) >= eps:
                print('Grad-Check Failed at (', i, ',', j, ') in output layer')
                print('Backpropagation:', output_W_grad[i, j], \
                      'Actual:', grad_calc)
                exit()
            
            # Restore original W value
            net.output_layer.W[i, j] += eps
            
    # Check output layer gradients for bias
    for i in range(net.output_layer.b.shape[0]):
        # Compute positive deviation cost
        net.output_layer.b[i, 0] += eps
        cost_pos = net.backward(net.forward(X), Y)
        
        # Compute the negative deviation cost
        net.output_layer.b[i, 0] -= 2 * eps
        cost_neg = net.backward(net.forward(X), Y)
        
        # Compute the numerical gradient
        grad_calc = (cost_pos - cost_neg) / (2 * eps)
        
        # Check if it is within tolerance limit
        if np.abs(grad_calc - output_b_grad[i, 0]) >= eps:
            print('Grad-Check Failed at', i, 'in output layer')
            print('Backpropagation:', output_b_grad[i, 0], \
                  'Actual:', grad_calc)
            exit()
        
        # Restore original b value
        net.output_layer.b[i, 0] += eps
        
    
    print('Grad-Check Successful')
