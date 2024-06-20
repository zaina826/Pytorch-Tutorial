#Tensor Basics
import torch
x=torch.rand(2,2,3,3)
x=torch.zeros(2,2,3,3)
x=torch.ones(2,2,3,3)
print(x.dtype)

x=torch.ones(2,2,dtype=torch.int) #Make the tensor a certain data type
print(x)
print(x.size())

x= torch.tensor([[0.1,0.24],[0.2,0.5]])
print(x)

x= torch.rand(2,2)
y= torch.rand(2,2)
z= torch.add(x,y) #Same with sub, mul and div
print("Z")
print(z)
#OR
y.add_(x) #Trailing underscore will do an inplace operation
print(y)

#Slicing
print("SLICING")
x= torch.rand(5,3)
print(x)
#Only the first column, all the rows
print(x[:,0])
#Only the second row, all the columns
print(x[1,:])
#Get a specific element
print(x[1,1])
#To get the actual value of one item exactly
print(x[1,1].item())

#Reshaping a tensor
x=torch.rand(4,4)
print(x)

y= x.view(16)
print(y)

#If you want the column/row number to be a specific number, but you're not sure what the other should be
#Use -1
y=x.view(-1,8)
print(y.size())
#You can see it has decided that the number of rows should be 2


#Numpy and Pytorch
import numpy as np
a=torch.ones(5)
print(a)
#Converts from tensor to numpy array
b=a.numpy()
print(b)
print(type(b))
#NOTE: This is a shallow copy, ie the same pointer, when one is changed so is the other
a.add_(1)
print(a)
print(b)

#The other way around
a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print(b) #By default float64

#Same goes here about the shallow copy
a+=1
print(a)
print(b)

#There's something called requires grad, means requires gradient. 
#Will be discussed in detail more later.
x= torch.ones(5,requires_grad=True)
print(x)