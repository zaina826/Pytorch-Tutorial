import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size, hidden_size) #linear function that maps from input size to hidden size
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size,num_classes)#linear function that maps from hidden size to output size which is the number of classes
    
    def forward(self,x):
        out= self.linear1(x)
        out= self.relu(out)
        out= self.linear2(out)
        #NO SOFTMAX JUST RAW
        return out
model= NeuralNet(28*28, 5,3)
criterion= nn.CrossEntropyLoss() #Applies Softmax

#For a binary problem, we used Sigmoid (also puts it into a probability)
#THEN: We must use nn.BCELoss()