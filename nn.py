import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def activation(z, derivative=False):
    if(derivative):
        return activation(z)*(1-activation(z))
    else:
        return 1.0/(1.0 + np.exp(-z))

def cost_function(y_true, y_pred):
    n = y_pred.shape[1]
    cost = (1./2*n)*np.sum((y_true - y_pred)**2)
    return cost

def cost_function_prime(y_true, y_pred):
    cost_prime = y_true - y_pred
    return cost_prime

class NeuralNet:

    def __init__(self, sizes):
        self.sizes = sizes
        self.no_of_layers = len(sizes)
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

    def forward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = activation(np.dot(w,a) + b)
        return a
    
    def SGD(self, train_data, epochs, batch_size, learning_rate):
        n = len(train_data)

        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[j:j+batch_size] for j in range(0, n, batch_size)]
            for batch in batches:
                self.update(batch, learning_rate)

    def update(self, batch,  learning_rate):
        n = len(batch)
        layer_b = [np.zeros(bias.shape) for bias in self.biases]
        layer_w = [np.zeros(weight.shape) for weight in self.weights]

        for x,y in batch:
            delta_layer_b, delta_layer_w = self.backprop(x, y)
            layer_b = [new_bias+delta_new_bias for new_bias, delta_new_bias in zip(layer_b, delta_layer_b)]
            layer_w = [new_weight+delta_new_weight for new_weight, delta_new_weight in zip(layer_w, delta_layer_w)]
        
        self.weights = [w - (learning_rate/n)*nw for w, nw in zip(layer_w, delta_layer_w)]
        self.biases = [b - (learning_rate/n)*nb for b, nb in zip(layer_b, delta_layer_b)]

    def backprop(self, x, y):
        layer_b = [np.zeros(bias.shape) for bias in self.biases]
        layer_w = [np.zeros(weight.shape) for weight in self.weights]

        a = x
        activations = [x]
        z_list = []
        for b,w in zip(self.biases, self.weights):
            z_list.append(np.dot(w,a) + b)
            a = activation(np.dot(w,a) + b)
            activations.append(a)

        delta = cost_function_prime(activations[-1], y)*activation(z_list[-1], derivative=True)
        layer_b[-1] = delta
        layer_w[-1] = np.dot(delta, activations[-2].T)

        for i in range(2, self.no_of_layers):
            delta = np.dot(self.weights[-i+1].T, delta)*activation(z_list[-i], derivative=True)
            layer_w[-i] = np.dot(delta, activations[-i-1].T)
            layer_b[-i] = delta
        return (layer_b, layer_w)

    def evaluate(self, test_data):
        results = [(np.argmax(self.forward(x)), y) for (x,y) in test_data]
        ans = sum(int(x==y) for (x,y) in results)
        return ans