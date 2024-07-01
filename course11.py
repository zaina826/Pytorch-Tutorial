import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x), axis=0)

x= np.array([2.0, 1.0,0.1])
outputs= softmax(x)
print("Softmax output: ", outputs)

#The sum of the outputs is one, all probabilities add up to one of course.


#To do this using pytorch
x= torch.tensor([2.0, 1.0,0.1])
outputs= torch.softmax(x, dim=0) #Computes the softmax across the first axis
print("Softmax with pytorch output: ", outputs)

#Cross-Entropy Loss
#The Softmax function combined to give us one number between 0 and 1 and it increases
#as the prediction diverges from the data.
#The better the prediction the lower the CEL
#It's done using two tensors:   
    #One of which will be Y in one-hot encoding
    #The other will be the tensor of predictions that we get from Softmax

def cross_entropy(actual, predicted):
    loss= -np.sum(actual*np.log(predicted)) #Formula
    #We can also normalize it here if we want
    return(loss)

Y= np.array([1,0,0])

Y_pred_good= np.array([0.9,0.1,0.2])
Y_pred_bad= np.array([0.1,0.5,0.3])

good_loss= cross_entropy(Y, Y_pred_good)
bad_loss= cross_entropy(Y, Y_pred_bad)

print(f'Good Loss = {good_loss:.4f}')
print(f'Bad Loss = {bad_loss:.4f}')


#To do this with just Pytorch
loss= nn.CrossEntropyLoss()
#CrossEntropyLoss() Already applies Softmax!
#So we must not implement the Softmax for out last layer
#i.e the last layer has just raw digits
#The Y class here is LABELED not one hot

Y=torch.tensor([0]) #So class zero, in one hot it would look like [1, 0, 0, ....]

#Predictions, no softmax
Y_pred_good=torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad=torch.tensor([[0.5,2.0,0.3]])

l1= loss(Y_pred_good, Y)
l2= loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

#So what would these predictions choose?
_, prediction1= torch.max(Y_pred_good,1) #Along the first axis
_, prediction2= torch.max(Y_pred_bad,1) #Along the first axis

print(f'Good Prediction chooses: {prediction1}')
print(f'Bad Prediction chooses: {prediction2}')


#For multiple samples:
#3 Samples with 3 possible classes
Y=torch.tensor([2,0,1])
#Prediction size = n_samples*n_classes= 3*3

#This means that for the first row the item with index two should be the highest, for the second row the iterm
#with index zero sould be the highest and for the last the item with index 1 should be the highest
predictions_good= torch.tensor([[0.1,1.0,2.1],[2.0, 1.0, 0.1],[0.1,3.0,0.1]])
predictions_bad= torch.tensor([[2.1,2.0,0.1],[0.2, 1.0, 0.6],[1.2,0.1,0.7]])

l1= loss(predictions_good, Y)
l2= loss(predictions_bad, Y)

print("Multi-Class")
print(l1.item())
print(l2.item())

_, prediction1= torch.max(predictions_good,1) #Along the first axis
_, prediction2= torch.max(predictions_bad,1) #Along the first axis

#The better prediction guesses the correct prediction, meanwhile the inferior one is just stupid.

print(f'Good Prediction chooses: {prediction1}')
print(f'Bad Prediction chooses: {prediction2}')

