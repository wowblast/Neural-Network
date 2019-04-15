
import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y,no_inputs,middle,outputs_no):
        self.no_inputs = no_inputs
        self.middle = middle
        self.outputs_no = outputs_no
        self.input      = x
        self.weights1   =  np.random.rand(self.input.shape[1],no_inputs) 
        self.weights2   = np.random.rand(no_inputs,middle) 
        self.outweight =np.array( [np.random.rand(middle,1) for _ in range(outputs_no)])
        #self.weights3   = np.random.rand(300,1) 
        #self.weights31   = np.random.rand(7,1)                 
        for l in  y:        
            self.y = np.array( [1 if x ==l else 0 for x in range(outputs_no)])
            

        self.output =np.array([np.zeros(self.y.shape) for x in range(outputs_no)])

       
        
       # self.output     = np.zeros(self.y.shape)
        #self.output2     = np.zeros(self.y2.shape)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1,self.weights2))
        for x in range(self.outputs_no):

            self.output[x] = sigmoid(np.dot(self.layer2, self.outweight[x]))
        #self.output2 = sigmoid(np.dot(self.layer2, self.weights31))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        for x in range(self.outputs_no):
            d_weights3 = np.dot(self.layer2.T, (2*(self.y[x]- self.output[x]) * sigmoid_derivative(self.output[x])))
            d_weights2 = np.dot(self.layer1.T,  (np.dot(2*(self.y[x] - self.output[x]) * sigmoid_derivative(self.output[x]), self.outweight[x].T) * sigmoid_derivative(self.layer2)))
            d_weights1 = np.dot(self.input.T,  (np.dot( (np.dot(2*(self.y[x] - self.output[x]) * sigmoid_derivative(self.output[x]), self.outweight[x].T) * sigmoid_derivative(self.layer2)), self.weights2.T) * sigmoid_derivative(self.layer1)))
            self.weights1 += d_weights1
            self.weights2 += d_weights2

        
            self.outweight[x]= d_weights3
           
        '''
        d_weights31 = np.dot(self.layer2.T, (2*(self.y2 - self.output2) * sigmoid_derivative(self.output2)))
        d_weights2 = np.dot(self.layer1.T,  (np.dot(2*(self.y2 - self.output2) * sigmoid_derivative(self.output2), self.weights31.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T,  (np.dot( (np.dot(2*(self.y2 - self.output2) * sigmoid_derivative(self.output2), self.weights31.T) * sigmoid_derivative(self.layer2)), self.weights2.T) * sigmoid_derivative(self.layer1)))
    

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights31 += d_weights31'''
def load_test(values):
    print ("files loades")


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
#my_neural_network.test(getTrainData()[0][1],getTrainData()[1][1])

if __name__ == "__main__":
    X = np.array([[0,0,1,0],
                  [0,1,1,0],
                  [1,0,1,0],
                  [1,1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    #X = np.array([getTrainData()[0][0] ])
    #y = np.array([getTrainData()[1][0] ])
    
    nn = NeuralNetwork(X,y,5,3,1)

    for i in range(1):
        nn.feedforward()
        nn.backprop()
    for x in range(10):
        print (nn.output[x],"\n")
   
    #print(nn.output)
    #print(nn.output2)



