import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#Device Configuration (same as feed-forward in prev course)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-parameters
num_epochs=4
batch_size=4
learning_rate=0.001

#dataset has PILImage of range [0,1] but we normalize to get range of [-1,1] using a transform
transform= transforms.Compose(
    #A composed transform first to tensor, then normalize
    [transforms.ToTensor,
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
     #This sets 0.5 as the mean, and 0.5 as also the standard deviation
     #so zero will become -1, 1 will become 1 and all other values
     #in between will get distributes from -1 to 1.
     #As for the 3 0.5s, each one of them is for one of the RGB colors.
)


train_dataset= torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset= torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader= torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)

classes=('plane', 'car', 'bird', 'cat','deer','dog','frog','horse','ship','truck')

class CNN(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

model= CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #4,3,32,32= 4,3,1024
        #Input channels and output channels
        images= images.to(device)
        labels= labels.to(device)

        #Forward Pass:
        outputs= model(images)
        loss=criterion(outputs,labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000==0:
            print(f'epoch {epoch+1} / {num_epochs}, step{i+1}/{n_total_steps}, loss= {loss.item():.4f}')

print("TRAINING DONE")

#Testing:
with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs= model(images)
        #max returns (values, index) we just want the index
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()

        for i in range(batch_size):
            label= labels[i]
            pred = predicted[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    acc = 100.0*n_correct/n_samples
    print(f'Accurcay of the network: {acc} %')

    for i in range(10):
        acc = 100.0*n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy for class {classes[i]} is : {acc}% ')

