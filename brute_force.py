#!/usr/bin/env python3

# Rough outline of brute-force binary classification
#   Issues: classification attempts to get ouput as 0 or 1, may not be clear cut
#   thresholding function might be required
#   Essentially- extracting inputs in form of k-mer count from sequence,
#       using single perceptron classification to attempt feature detection
#   Next build- attempt two perceptrons for output detection- one for each species   

import numpy as np

class network:
    def __init__(self):
        self.hidden = [ 0 for i in range((4**10)+1)]
        self.output = 0
        self.bias = 0
        self.weights = [ 1/(4**10) for i in range((4**10)+1)]
        self.error = 0
        self.delta = 0
        self.delta_weights = [0 for i in range((4**10)+1)]

    def feed_forward(self, sequence):
        for i in range(len(sequence)-9):
            add_index = get_index(sequence[i:i+10])
            self.hidden[add_index] += 1
        for i in range((4**10)+1): output += self.hidden[i] * self.weights[i]
        self.output = 1/(1+ np.exp(-output))

    def back_propagation(self, species):
        if species == "HIV": target = 1
        else: target = 0
        self.error = target - self.output
        self.delta = self.error * (1 - self.output) * self.output

    def update_weights(self, eta):
        self.bias = slef.bias + self.delta * eta
        self.delta_weights = [ self.delta * eta * self.hidden[i] for i in range((4**10)+1)]
        self.weights = [ self.weights[i] + self.delta_weights[i] for i in range((4**10)+1)]

    def report(self,sequence):
        self.feed_forward(sequence)
        return self.output

def get_index(sequence):
    index = 0
    for i in range(len(sequence)):
        token = sequence[len(sequence)-1-i:len(sequence)-i]
        if token == "A": index += (0 * (4**i))
        elif token == "C": index += (1 * (4**i))
        elif token == "G": index += (2 * (4**i))
        else: index += (3 * (4**i))
    return index
