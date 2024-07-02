#If we were to just have linear transitions between each layer
#The whole network could be reduced to a linear regression model
#With just one equation.
#But this is usually not good enough, so by adding some non-linear
#functions to our model, it can learn better and preform more complex tasks

#Some examples of these non-linear (Activation) functions:

#1)Step Function:
#The step function returns 1 if the input is greater than a threshold
#and zero otherwise. Not used in practice.

#2) Sigmoid Function: We talked about it before
# Outputs a probability between 0 and 1, used in the last layer of a binary classification
#function 

#3) Hyperbolic Tangent Function : Scaled Sigmoid funciton, shifted
#Outputs a value between -1 and 1, a good choice in hidden layers

#4) RELU Function: The most popular choice 
# Outputs zero for negative values, outputs input as it is for positive values
# If you don't know which function to use for hidden layers, use ReLU

#5) Leaky ReLU: Modified and improved ReLU:
#Same for positive numbers, mutliply our input 
#For negative numbers, multiplies them by a very small value like 0.001
#This tries to solve the vanishing gradient problem 
#If the gradient is zero, the neurons won't be updated so those neurons won't learn anything (Dead Neurons)


#6) Softmax : Already mentioned
#Last layer of a multi-class classification problem

import torch
import torch.nn as nn

#Option 1: Define your own functions and then call them
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size): 
        #Define the functions:
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size, hidden_size)
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size,1)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        #Call the functions for forward propagation
        out= self.linear1(x)
        out= self.relu(out)
        out= self.linear2(out)
        out= self.sigmoid(out)
        return out
    
#Option 1: Call them straight away in the forward propagation function
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size): 
        #Define the functions:
        super(NeuralNet,self).__init__()
        #Only defining the linear layer
        self.linear1=nn.Linear(input_size, hidden_size)
        self.linear2= nn.Linear(hidden_size,1)

    def forward(self,x):
        #Call the functions for forward propagation
        out= torch.relu(self.linear1(x))
        out= torch.sigmoid(self.linear2(out))
        return out
    
#Relu can be replaced with any other activation function of the ones we mentioned above
#Sometimes they're not available in torch api directly, but can be available through this:
import torch.nn.functional as F
#Leaky ReLU for example is only available there.
    