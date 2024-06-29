# Epoch : 1 forward and 1 backward pass over all training samples
#    During an Epoch the model sees every training sample once.

# Batch Size: Number of training samples in one forward and backward pass
#    The batch size is the number of training samples processed before the model's internal parameters are updated.
#    Training with a batch size allows you to use a subset of the dataset for each forward and backward pass, which can be more efficient, especially with large datasets.

# Number of iterations : number of passes, each pass using [batch size] training samples.
#   The number of iterations is the number of batches needed to complete one epoch.
#   It is calculated as the total number of training samples divided by the batch size.
#   For example, if you have 100 samples and a batch size of 20, you will have 5 iterations per epoch (100/20 = 5).

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class Wine(Dataset): #This means it inherits Dataset
    def __init__(self):
        xy= np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1) #Skip rows to skip the header
        self.x= torch.from_numpy(xy[:,1:]) #Skip the first column, cuz that's the y
        self.y= torch.from_numpy(xy[:,[0]]) #We have kinds 1, 2 and 3 , this is going to be of dim n_samples, 1
        self.num_samples= xy.shape[0] 
        #Load the data

    def __getitem__(self, index):
        return self.x[index], self.y[index]
        #dataset[0]
    def __len__(self):
        return self.num_samples
        #len(dataset)

dataset= Wine()

first_data= dataset[0] 
features, labels= first_data
print(features)
print(labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
dataiter= iter(dataloader)
data = next(dataiter)  #This will give us the next batch
features, labels= data
print(features)
print(labels)

#Let's try doing a training loop :
num_epochs= 2
total_samples= len(dataset)
n_iterations = math.ceil(total_samples/4) #Remember 4 is the batch size
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(dataloader):
        #The enumerate will give us the index and the unpacked inputs and labels
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step{i+1}/{n_iterations}, inputs {inputs.shape}')