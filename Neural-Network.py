
import numpy as np
import random as rd

class Perceptron:

    def __init__(self, no_of_inputs, learning_rate):
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.weights = np.array ([0.0+ (rd.randint(-3, 3))/10.0 for _ in range(no_of_inputs+1) ])
        self.output = 0
        self.error = 0
    def activation(self,inputs):
        self.output = np.dot(inputs, self.weights[1:]) +self.weights[0]
         
        return sigmoid( self.output)
    
    '''
    def train(self, inputs, targets):
                
             
        for input_, target in zip(inputs,targets):
            activation = self.test(input_)

            self.weights[1:] += self.learning_rate *(target - activation) *np.asarray(input_)
            self.weights[0] += float(self.learning_rate * (target - activation))
    def __str__(self):
        print (self.weights)  '''      

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
        targets = [1  if x == 5-1 else 0  for x in range(10)]
        #input
        for x in range(len(self.inputlayer)):
            middle_input.append(self.inputlayer[x].activation(inputs[x])[0] )
        
        

        for x in range(len(self.hiddenlayer) ):
            output_input.append(self.hiddenlayer[x].activation(np.array((middle_input))))    
        for x in range(len(self.outputlayer) ):
            ouputs.append(self.outputlayer[x].activation(np.array(output_input)))
        print (ouputs)            
        return ouputs
       
    def back_propagation(self,inputs,target):
        outputs = self.forward_propagation(inputs,target) 
        self.update_weights_outputs_layers(outputs,target)
        self.update_weights_hidden_layers()
        self.update_weights_inputs_layers()
        #error hiden-outpt = o(1-o)(target-o)   
        # erro inut-hidden = h (1-h) (error out)        
        #outerror = outputs[0]*(1-outputs[0])*(target/10-outputs[0])
    def update_weights_outputs_layers(self,outputs,target):
        outerror = outputs[0]*(1-outputs[0])*(target/10-outputs[0])

        for x in range(len(self.outputlayer)):
            self.outputlayer[x].weights + [self.learning_rate*outerror*self.outputlayer[x].output for _ in range( len(self.outputlayer[x].weights)) ]
        for x in range(len(self.hiddenlayer)):

            self.hiddenlayer[x].error =  self.hiddenlayer[x].output*(1-self.hiddenlayer[x].output)*outerror
    def update_weights_hidden_layers(self):

        for x in range(len(self.hiddenlayer)):
            self.hiddenlayer[x].weights + [self.learning_rate*self.hiddenlayer[x].error*self.hiddenlayer[x].output for _ in range( len(self.hiddenlayer[x].weights)) ]
            for z in range(len(self.inputlayer)):

                self.inputlayer[z].error =  self.inputlayer[z].output*(1-self.inputlayer[z].output)*self.hiddenlayer[x].error  
    def update_weights_inputs_layers(self):
        for x in range(len(self.inputlayer)):
            self.inputlayer[x].weights + [self.learning_rate*self.inputlayer[x].error*self.inputlayer[x].output for _ in range( len(self.inputlayer[x].weights)) ]


def load_test(values):
    print ("files loades")

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

print (getTestData()[1][0])
#print (getTestData()[0][0])

listtrain = [[0.0,0.0],[0.0,1.0],[1.0,0.0]]     
my_p = Perceptron(2,0.01)
#my_p.__str__()

no_inputs=28*28
ind_learning_rate=1
my_neural_network = Neural_network(no_inputs,28,10,ind_learning_rate,1)
for x in my_neural_network.outputlayer:
    print x.weights[0]

my_neural_network.back_propagation(getTestData()[0][0],getTestData()[1][0])
for x in my_neural_network.outputlayer:
    print x.weights[0]
#my_neural_network.forward_propagation(getTestData()[0][0],getTestData()[1][0])

#my_p.train(listtrain,[0.0,1.0,1.0])
#my_p.__str__()
#rint (my_p.test([0,0]))
ls =[1,2]
ls1 = [1,2]
#print (np.array (ls1) +np.array(ls)) 

