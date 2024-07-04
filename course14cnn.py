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
    [transforms.ToTensor(),
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
        super(CNN, self).__init__()
        #(input channel size, output channel size)
        #initially the shape is 4, 3, 32, 32
        self.conv1 = nn.Conv2d(3,6,5) 
        #After the first convultion since we don't have padding, it will become 
        #4,6,28,28
        #6 Output channels (each one of them done by a certain kernel)
        #Each kernel is of size 5  
        #We don't actually decide what the kernel does or what it looks like
        #These are parameters just like weights and biases that change through out the training.

        self.pool= nn.MaxPool2d(2,2) #Kernel size of 2, and a stride of
        #Now pooling by a kernel of size 2 will divide the pixels in half, so 4,6,14,14
        self.conv2= nn.Conv2d(6,16,5)
        #4,16,10,10
        #Now if we apply pooling again: 4,16,5,5
        #so this is why the first linear layer must have an input of 16*5*5
        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) #First conv then pooling
        x=self.pool(F.relu(self.conv2(x))) #Second
        #This is called Flattening, so we can plug it into a linear function
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x) #No activation function, cuz softmax is included in the loss
        return x

model= CNN().to(device)
criterion=nn.CrossEntropyLoss() #Remember softmax is already included here
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

