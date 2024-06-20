#Autograd package and how to calculate gradients with it
# #Gradients are essential for all optimization
# Meaning:

# The gradients in x.grad represent how z_mean (the mean of z) changes with respect to changes in x.
# Specifically, each value in x.grad tells us how much z_mean would increase if we increased the corresponding value in x by a tiny amount.
# For example, if x_1 increases slightly, z_mean would increase by approximately (The gradient of z with respect to x1) 2.0 times that small amount.

# Gradients:
# The gradients of x are the derivatives of the loss (z_mean in this case) with respect to x.
# They indicate how sensitive the loss is to changes in x.
# In training a neural network, these gradients are used to update the model parameters to minimize the loss.

import torch
x = torch.randn(3, requires_grad=True) #requires_grad is false by default
print(x)

y=x+2
#Since x has requires_grad=True, the tensor y will also have requires_grad=True and track the gradient.
print(y)

z=y*y*2
print(z)

#You can do this, which gives z as a scalar (The mean), when the loss is a scalar value
# z=z.mean()
# z.backward()
# print(z)

#OR 
#When you pass a parameter (vector) to backward(),
#  you are telling PyTorch to compute the gradient of a 
# vector-Jacobian product instead of just the gradients of a
#  scalar function. This is useful when z is not a scalar.
#Used for more advances operations, such as computing gradients for a custom scalar function or when z is a non-scalar tensor.
v=torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
z.backward(v) #Finds dz/dx using the chain rule (Jacobian Product)
print(x.grad)
#Remember, this gradient tells us how small changes in x affect z (The final output).


f= torch.randn(3,requires_grad=True)
print(f)
#To prevent gradient history
f.requires_grad_(False)
print("Falsify")
print(f)

#OR
f= torch.randn(3,requires_grad=True)
y=f.detach()
print("Detach")
print(y)

#OR
print("With with")
with torch.no_grad():
    d=f+2
print(d)

#EXAMPLE
weights= torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad)

    #So they don't accumilate. We must reset them before every epoch
    weights.grad.zero_()

#Will be mentioned in depth in later tutorials
# otherweights= torch.ones(4, requires_grad=True)
# optimizer = torch.optim.SGD(weights, lr=0.01) #Learning rate
# optimizer.step()
# optimizer.zero_grad()
