
import numpy as np
import random as rd

import datetime as dt
from multiprocessing import Process, current_process


class Perceptron:

    def __init__(self, no_of_inputs, learning_rate):
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.weights = np.array ([0.0+ (rd.randint(-3, 3))/10.0 for _ in range(no_of_inputs+1) ])
        self.output = 0
        self.error = 0
    def activation(self,inputs):
        self.output = sigmoid( np.sum (np.dot(inputs, self.weights[1:]) +self.weights[0]))
        return  self.output
    
    

class Neural_network:
    def __init__(self,inputlayer,hiddenlayer,outputlayer,learning_rate,epochs):
        self.inputlayer = [Perceptron(1,learning_rate) for _ in range(inputlayer)]
        self.hiddenlayer = [Perceptron(inputlayer,learning_rate) for _ in range(hiddenlayer)]
        self.outputlayer = [Perceptron(hiddenlayer,learning_rate) for _ in range(outputlayer)]
        self.learning_rate = learning_rate
        self.epochs = epochs
    def forward_propagation(self,inputs,target):
        middle_input = []
        output_input =[ ]
        ouputs = []
        #input
        for x in range(len(self.inputlayer)):
            
            middle_input.append(self.inputlayer[x].activation(inputs[x]))
            
        
        

        for x in range(len(self.hiddenlayer) ):
            output_input.append(self.hiddenlayer[x].activation(middle_input))    
        for x in range(len(self.outputlayer) ):
            ouputs.append(self.outputlayer[x].activation(np.array(output_input)))
        return ouputs

    def test(self,inputs,target):
        middle_input = []
        output_input =[ ]
        ouputs = []
        #input
        for x in range(len(self.inputlayer)):
            
            middle_input.append(self.inputlayer[x].activation(inputs[x]))
            
        
        

        for x in range(len(self.hiddenlayer) ):
            output_input.append(self.hiddenlayer[x].activation(middle_input))    
        for x in range(len(self.outputlayer) ):
            ouputs.append(self.outputlayer[x].activation(np.array(output_input)))
        return ouputs    
       
    def back_propagation(self,inputs,target):
        outputs = self.forward_propagation(inputs,target) 
        self.update_weights_outputs_layers(outputs,target)
        
        #error hiden-outpt = o(1-o)(target-o)   
        # erro inut-hidden = h (1-h) (error out)        
        #outerror = outputs[0]*(1-outputs[0])*(target/10-outputs[0])
    #updte final weigths    
    def update_weights_outputs_layers(self,outputs,target):
        outerror = outputs[0]*(1-outputs[0])*(target/10-outputs[0])
        targets = [1 if x ==target else 0 for x in range(len(self.outputlayer))]
        targets = target
        for x in range(len(self.outputlayer)):
            outerror = outputs[0]*(1-outputs[0])*(target/10-outputs[0])
            self.outputlayer[x].weights = self.outputlayer[x].weights + [self.learning_rate*outerror*self.outputlayer[x].output for _ in range( len(self.outputlayer[x].weights)) ]
            for x in range(len(self.hiddenlayer)):

                self.hiddenlayer[x].error =  self.hiddenlayer[x].output*(1-self.hiddenlayer[x].output)*outerror
            self.update_weights_hidden_layers()
            self.update_weights_inputs_layers()    
    #update hidden weigths        
    def update_weights_hidden_layers(self):

        for x in range(len(self.hiddenlayer)):
            self.hiddenlayer[x].weights= self.hiddenlayer[x].weights + [self.learning_rate*self.hiddenlayer[x].error*self.hiddenlayer[x].output for _ in range( len(self.hiddenlayer[x].weights)) ]
            for z in range(len(self.inputlayer)):

                self.inputlayer[z].error =  self.inputlayer[z].output*(1-self.inputlayer[z].output)*self.hiddenlayer[x].error  
    #update inputs weigths            
    def update_weights_inputs_layers(self):
        for x in range(len(self.inputlayer)):
            self.inputlayer[x].weights= self.inputlayer[x].weights + [self.learning_rate*self.inputlayer[x].error*self.inputlayer[x].output for _ in range( len(self.inputlayer[x].weights)) ]



def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
def getTrainData():
    return getData("mnist_train.npz")

def getTestData():
    return getData("mnist_test.npz")

def getValidationData():
    return getData("mnist_valid.npz")

def getData(fname):
    with np.load(fname) as data:
        images = data['images']
        labels = data['labels']
    return images, labels






no_inputs=28*28
ind_learning_rate=0.02
my_neural_network = Neural_network(no_inputs,380,10,ind_learning_rate,1)






X = np.array(getTrainData()[0])
y = np.array(getTrainData()[1])
for _ in range(1):
    for x in range(1):
        my_neural_network.back_propagation(X[x],y[x])
  
pred = []
real = []        
for x in range(1):
    for outp in my_neural_network.test(X[x],y[x]):
        print (outp)
    


print ("trainig complete")

