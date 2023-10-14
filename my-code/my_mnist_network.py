import numpy as np
import random
import json

class Network:
    def __init__(self, sizes) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # a list of 1d arrays 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # a list of 2d matricies. j, k still follows w_j,k still is the weight to neuron j from neuron k.
        self.cost_function = "Quadratic Cost"
    
    def feedforward(self, a): # a for "activations", i.e. inputs
        """returns the output of the network for input a"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self , training_data , epochs , mini_batch_size , eta , test_data=None):
        if test_data:
            n_test = len(list(test_data))

        n = len(list(training_data))
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
        nabla_w = [np.zeros(weights.shape) for weights in self.weights] # this has the same shape as self.weights, but zeros.
        nabla_b = [np.zeros(biases.shape) for biases in self.biases]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # returns partial derivatives for all weights and biases in the same format as self.weights and self.biases
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # adding two matricies together, cumulative sum 
        
        # gradient descent b -> b' = b - eta* nabla C 
        # we divide eta by the nunmber of training examples to get an average
        self.biases = [biases - eta*nb/len(mini_batch) for biases, nb in zip(self.biases, nabla_b)]
        self.weights = [weights - eta*nw/len(mini_batch) for weights, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        activations = [x]
        weighted_inputs = [] # this has one less element than `activations`
        nabla_biases = [np.zeros(biases.shape) for biases in self.biases]
        nabla_weights = [np.zeros(weights.shape) for weights in self.weights]

        # feedforward and record values of all activations. Also store weighted inputs for the output layer, L.
        for weights, biases in zip(self.weights, self.biases):
            activation = activations[-1] # activations in layer l-1
            weighted_inputs_next = np.dot(weights, activation) + biases # weighted inputs to layer l
            activation_next = sigmoid(weighted_inputs_next) # activations to layer l
            
            # add the values of the next layer to the lists
            activations.append(activation_next)
            weighted_inputs.append(weighted_inputs_next)

        # compute delta values for the output layer with delta = dC/da * sigmoid-prime(z)
        delta_output = self.cost_derivative(activations[-1], y) * sigmoid_prime(weighted_inputs[-1]) # delta values for the output layer
        nabla_biases[-1] = delta_output
        nabla_weights[-1] = np.matmul(delta_output, activations[-2].transpose()) # column vector * row vector = matrix

        # compute delta values for all other neurons
        delta = delta_output

        # we calculate the partial derivatives of each remaining layer here 
        for layer in range(len(self.weights) - 2, -1, -1): # the indexes of all layers excluding the first and last, counting backwards. 
            weights_vector = self.weights[layer + 1].transpose()
            delta = np.matmul(weights_vector, delta) * sigmoid_prime(weighted_inputs[layer])
            nabla_biases[layer] = delta
            nabla_weights[layer] = np.matmul(delta, activations[layer].transpose()) # these weights connect layer `layer` to layer `layer` + 1. This is the opposite to the w_j,k notation. So the index of the activation neurons is actually the same as the index of the weight matrix!

        return (nabla_biases, nabla_weights)

    

    def evaluate(self, test_data):
        """returns how many examples of test data the network got correct. We say the network chooses the digit with whose neuron has highest activation"""
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum([int(x==y) for x, y in test_results])

    def cost_derivative(self, output_activations, y):
        """returns a vector of partial derivatives dC/da where a is an output neuron, for every a"""
        return (output_activations - y)
    
    def save(self, filename):
        data = {"sizes" : self.sizes,
            "weights" : [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost" : self.cost_function}

        with open(filename, "w") as f:
            json.dump(data, f)

    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]



#### Miscellaneous functions

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))