import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
#Transforms: Transform numpy arrays or images into tensors so we can use them in neural networks


#From last tutorial:
class Wine(Dataset): #This means it inherits Dataset
    def __init__(self, transform = None):
        xy= np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1) #Skip rows to skip the header
        self.x= xy[:,1:] 
        self.y= xy[:,[0]]
        #NOW WE'RE NOT TURNING IT TO A TENSOR 

        self.num_samples= xy.shape[0] 
        self.transform = transform
        #Load the data

    def __getitem__(self, index):
        sample= self.x[index], self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return(sample)
        #dataset[0]
    def __len__(self):
        return self.num_samples
        #len(dataset)

class toTensor:
    def __call__(self, sample): #Default function of the class
        inputs, targets= sample
        return(torch.from_numpy(inputs), torch.from_numpy(targets))
    
dataset= Wine(transform=toTensor()) #Calls the Default function
first = dataset[0]
inputs, target = first
print(inputs)
print(type(inputs))
print(type(target))

#We notice they're of type Torch, which isn't true if we delte toTensor() and replace it with None.

#A mutliplication tranform
class mulTransform:
    def __init__(self, factor):
        self.factor= factor
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return(inputs, targets)

#A composed transform does multiple transforms on the data
composed = torchvision.transforms.Compose ({toTensor(), mulTransform(2)})
dataset = Wine(transform=composed)
first = dataset[0]
inputs, target = first
print(inputs)
print(type(inputs))
print(type(target))
