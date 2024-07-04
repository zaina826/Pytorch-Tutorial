import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#Device Configuration
#This is actually not really doing anything on the M2 chip, because it doesn't
#utilize an Nvidia GPU with Cuda, which is a parallel computational platform
#and is very helpful for running these neural networks fast.
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#The Jetson Nano actually has CUDA
#But the M2 chip does offer multicore processing, hyper threading and Advanced Vector Extensions
#Which can make our neural network run fast and be more accurate and precise.

#Hyper Parameters:
input_size= 784 #28 by 28 pixels
hidden= 100
num_classes= 10 #The output
num_epochs= 2 #How many times to iterate over the dataset
batch_size= 100 #The number of samples processed before updating the model parameters.
learning_rate=0.001

#Import MNIST dataset
train_dataset= torchvision.datasets.MNIST(root='./data',train= True,
                                         transform=transforms.ToTensor(),download=True )
test_dataset= torchvision.datasets.MNIST(root='./data',train= False,
                                         transform=transforms.ToTensor())

train_loader= torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) #Shuffle doesn't matter for the evaluation

examples = iter(train_loader)
samples, labels = next(examples) #next is called like this in the newer version I guess?
print(samples.shape, labels.shape)
#100 samples, 1 channel (no colors), 28, 28
#Lables: tensor of size 100

#Let's take a look at some of the images
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

#Now we create the neural network:
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        #Start creating layers
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu= nn.ReLU()
        self.l2= nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out= self.l1(x)
        out= self.relu(out)
        out= self.l2(out)
        #Don't apply softmax since we'll use CEL, which applies Softmax
        return out
    
model= NeuralNetwork(input_size, hidden, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate)
#Adam (short for Adaptive Moment Estimation) is an optimization algorithm used in training machine learning models, especially deep neural networks. It combines the advantages of two other popular optimization methods: AdaGrad and RMSProp.

#Training Loop
n_total_steps= len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #100,1,28,28
        images= images.reshape(-1, 28*28).to(device)
        labels= labels.to(device)

        #Forward:
        outputs= model(images)
        loss= criterion(outputs, labels)

        #Backward: 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print(f'epoch {epoch+1} / {num_epochs}, step{i+1}/{n_total_steps}, loss= {loss.item():.4f}')

#test
with torch.no_grad(): 
    n_correct= 0
    n_samples = 0
    for images, labels in test_loader:
        images= images.reshape(-1, 28*28).to(device)
        labels= labels.to(device)
        outputs= model(images)

        _,predictions= torch.max(outputs,1) #Returns the value with the highest probability, along the first axis 
        n_samples+=labels.shape[0]
        n_correct+= (predictions==labels).sum().item()

    accuracy= 100.0*n_correct/n_samples
    print(f'accuracy : {accuracy}')