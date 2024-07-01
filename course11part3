import torch
import torch.nn as nn

class NeuralNetBinary(nn.Module):
    def __init__(self,input_size, hidden_size): #Num classes is 1 now
        super(NeuralNetBinary,self).__init__()
        self.linear1=nn.Linear(input_size, hidden_size) #linear function that maps from input size to hidden size
        self.relu= nn.ReLU()
        self.linear2= nn.Linear(hidden_size,1)#linear function that maps from hidden size to output size which is the number of classes
    
    def forward(self,x):
        out= self.linear1(x)
        out= self.relu(out)
        out= self.linear2(out)
        #Sigmoid:
        out= torch.sigmoid(out)
        return out
model= NeuralNetBinary(28*28, 5)
criterion= nn.BCELoss() #Applies Softmax

#Like course 10 I think