#The video explains well the process of finding the gradient vector
#using the loss function, which is usually mean squared error.

#Now after we actually find this gradient, we will need to update our
#neural network to actually learn, this isn't discussed in this video
#but here's some context on how to actually do it.

# Using the Gradient: Gradient Descent Algorithm
# The gradient descent algorithm uses the gradient to iteratively adjust the weight in order to minimize the loss function. The update rule for the weight is typically:
# new weight =old weight−𝜂⋅∂loss/∂weight

#So let's take an example on finding the gradient:
import torch
x=torch.tensor(1.0) #Input
y=torch.tensor(2.0) #Target Output
w=torch.tensor(1.0, requires_grad=True) #W here represents weight

#Forward pass and compute the loss
y_hat= w*x #This is the function that calculates y (The network's guess)
loss = (y_hat-y)**2 #Compute MSE
print("Y hat", y_hat)
print(loss)

#Backward Pass
loss.backward()
print(w.grad)

#So it says -2

#The magnitude implies that this weight has double the effect that 
# a weight with gradient 1 has for example.

#On the other hand a gradient of 4 for another weight is going to be 
#Double as important as this one.

#As for the magnitude, this means that the gradient descent, should 
#be in the opposite direction, so in the direction of -2, the exact 
#change that we would make to this weight, would be 

#Weight = Weight - (n*gradient)
#We keep doing this until convergence, when the weights become very small
#we say our networks have converged.


