import numpy as np
import math
import matplotlib.pyplot as plt
#BOTH FUNCTIONS ONLY FOR VERTICAL MATRICES
def sigmoid(X):
    S = np.zeros(shape=X.shape)
    for i in range(0,len(X)):
        S[i] = 1/(1+math.exp(-X[i][0]))
    return S
def sigmoidDer(X):
    S = np.zeros(shape=X.shape)
    S = sigmoid(X)*(1-sigmoid(X))
    return S 
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size,input_size)
        #self.weights = np.ascontiguousarray(self.weights)
        self.biases = np.random.randn(output_size,1)
    def Forward(self, inputs):
        self.remInputs = inputs
        self.S = self.weights @ inputs + self.biases
        self.Y = sigmoid(self.S)
        return self.Y
    def Backward(self, last_delta, last_weights):
        dervSig = sigmoidDer(self.S)
        wage_l_delta_prod = last_weights.T @ last_delta
        self.delta = dervSig * wage_l_delta_prod
    def Learn(self, learning_rate):
        if self.delta is None:
            raise Exception("THERE IS NO DELTA! Throw Backward before Learn")
        else:
            deltaWeight = self.remInputs @ self.delta.T
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*self.delta
class Model:
    def __init__(self, input_size, layer_sizes):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layers = []
        self.error = 0
        for i in range(0, len(self.layer_sizes)):
            if i == 0:
                layer = DenseLayer(input_size,layer_sizes[0])
            else:
                layer = DenseLayer(layer_sizes[i-1],layer_sizes[i])
            self.layers.append(layer)
    def Forward(self, inputs):
        '''Inputs->outputs'''
        input_data = np.expand_dims(np.array(inputs),0)
        if len(input_data[0]) != self.input_size:
            raise Exception("Wrong input size!")
        x = input_data.T
        for i in range(0, len(self.layer_sizes)):
            x = self.layers[i].Forward(x) 
        return x
    def Backward(self, inputs, real_values):
        '''Calculating deltas and error'''
        #first delta - the 1D matrix containing all first derivtives of target function via model-output values
        predictions = self.Forward(inputs)
        self.error = math.sqrt(np.sum((predictions - real_values.T)*(predictions - real_values.T)))

        #print(predictions, real_values.T)
        sq_delta = 2*(predictions - real_values.T) #Squared error
        delta = sigmoidDer( self.layers[len(self.layer_sizes)-1].S)*sq_delta
        self.layers[len(self.layer_sizes)-1].delta = delta
        for i in range(len(self.layer_sizes)-2, -1,-1):
            self.layers[i].Backward(delta, self.layers[i+1].weights)
            delta = self.layers[i].delta
    def Learn(self, learning_rate):
        '''Using deltas to learn the net'''
        for i in range(0, len(self.layer_sizes)):
            self.layers[i].Learn(learning_rate)
    def Train(self, data_base, epochs, learning_rate):
        '''Actual train function'''
        for i in range(epochs):
            cum_error = 0
            for data in data_base:
                input_data = data[0]
                output_data = np.expand_dims(np.array(data[1]),0)
                self.Backward(input_data,output_data)
                self.Learn(learning_rate)
                cum_error = cum_error + self.error
            cum_error = cum_error / len(data_base)
            print('Epoch nr. ' + str(i) + ' Error: ' + str(round(cum_error,4)))


#Sigmoid wiec wyjscia sa od [0,1]
