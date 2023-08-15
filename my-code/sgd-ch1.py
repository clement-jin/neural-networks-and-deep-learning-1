import numpy as np
import random
import matplotlib.pyplot as plt

class Network:
    def __init__(self, sizes) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # a list of 1d arrays 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # a list of 2d matricies
    
    def feedforward(self, a): # a for "activations", i.e. inputs
        """returns the output of the network for input a"""
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a) + np.transpose(b)[0]
        return a
    
    def SGD(self , training_data , epochs , mini_batch_size , eta , test_data=None):
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        
            if test_data:
                print(f"{epoch} epochs complete: {self.evaluate(test_data)}/{n_test}")
            else:
                print(f"{epoch} epochs complete.")

    def update_mini_batch(self, mini_batch, eta):
        """updates weights and biases according to the gradient calculated from the mini-batch."""
        pass

    def backprop(self, x, y):
        pass

    def evaluate(self, test_data):
        pass

    def cost_derivative(self, output_activations, y):
        pass



#### Miscellaneous functions

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))



# tests are here because I'm practical.



net = Network([2, 3, 3])
# o = net.feedforward(np.array([1, 1]))
# print(o)

# plt.plot(net.sigmoid(np.linspace(-5, 5, 50)))
# plt.show()


