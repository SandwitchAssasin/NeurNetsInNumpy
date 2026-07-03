import numpy as np
import math
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
            #print("KJA", self.weights.T.shape, self.remInputs, self.delta.T)
            deltaWeight = self.remInputs @ self.delta.T
            wT = self.weights.T - learning_rate*deltaWeight
            self.weights = wT.T
            self.biases = self.biases - learning_rate*self.delta
class Model:
    def __init__(self, input_size, layer_sizes):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(0, len(self.layer_sizes)):
            if i == 0:
                layer = DenseLayer(input_size,layer_sizes[0])
            else:
                layer = DenseLayer(layer_sizes[i-1],layer_sizes[i])
            self.layers.append(layer)
    def Forward(self, inputs):
        if len(inputs) != self.input_size:
            raise Exception("Wrong input size!")
        x = inputs
        for i in range(0, len(self.layer_sizes)):
            x = self.layers[i].Forward(x) 
        return x
    def Backward(self, inputs, real_values, learning_rate):
        #first delta - the 1D matrix containing all first derivtives of target function via model-output values
        predictions = self.Forward(inputs)
        sq_delta = 2*(predictions - real_values) #Squared error
        delta = sigmoidDer( self.layers[len(self.layer_sizes)-1].S)*sq_delta
        self.layers[len(self.layer_sizes)-1].delta = delta
        for i in range(len(self.layer_sizes)-2, -1,-1):
            self.layers[i].Backward(delta, self.layers[i+1].weights)
            delta = self.layers[i].delta
        self.Learn(learning_rate)
    def Learn(self, learning_rate):
        for i in range(0, len(self.layer_sizes)):
            #print(self.layers[i].output_size)
            self.layers[i].Learn(learning_rate)

m = Model(2, (4,6,1))
databs = [[(4,2),1],[(1,1),1],[(1.5,1.6),1],[(1.2,1.4),1],
          [(3.4,4),0],[(0.3,0.5),0],[(1.9,0.1),0],[(0.2,4.4),0],[(1.5,3.4),0],
          [(4,1.4),0],[(4,3),0]]
for dt in databs:
    dt[1] = [dt[1]]
print(m.Forward([[3],[4]]))
#Sigmoid wiec wyjscia sa od [0,1]
m.Backward([[3],[4]],[[1]], 0.02)
print(m.Forward([[3],[4]]))
