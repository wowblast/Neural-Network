
import numpy as np
import random as rd
class Perceptron():

    def __init__(self, no_of_inputs, learning_rate):
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.weights = [1.0+ (rd.randint(-3, 3))/10.0 for _ in range(no_of_inputs+1) ]
    def test(self,inputs):
        activation = np.dot(inputs, self.weights[1:]) 
        if activation > -(self.weights[0]):
           return 1
        else:
          return 0 
       
    def train(self, inputs, targets):
                
             
        for input_, target in zip(inputs,targets):
            activation = self.test(input_)

            self.weights[1:] += self.learning_rate *(target - activation) *np.asarray(input_)
            self.weights[0] += float(self.learning_rate * (target - activation))
    def __str__(self):
        print (self.weights)        
            
def load_test(values):
    print ("files loades")

listtrain = [[0.0,0.0],[0.0,1.0],[1.0,0.0]]     
my_p = Perceptron(2,0.01)
my_p.__str__()


my_p.train(listtrain,[0.0,1.0,1.0])
my_p.__str__()
print (my_p.test([0,0]))

