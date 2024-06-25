#Note about multiple layered networks. Usually we find the gradient for the last
#layer, directly from the loss function.
#Now we can change the last layer (the output layer), by changing the
#weights and biases, which is just the easy one level gradient.
#We can also change it by changing the activation of the layer
#before the last, which we can't do directly.
#But we can do this by changing the weights and biases of the second to last layer
#So according to the last layer, for every neuron, we sum up 
#The changes that we want to make to the  activation
#and pass that back to the second to last layer, so this way "we can
# propagate the change", by recursivly doing the same thing.
#This is very well explained in 3Blue1Brown Deep Learning, Chapter 3.

#Now usually we don't do this training example one by one.
#We use mini-batch stochastic gradient descent (SGD).
#In which we take a batch of say 100 training examples.
#Then we basically do this: forward propagation on each one, and calculate
#the loss for each one, then average all the 100 losses.
#Then consider that our loss, and do backpropagation using that loss.
#This will help us be more effecient and faster to converge after 
#multiple batches. Note these mini batches are not the same as epochs.


#Let's start on tutorial 5 from the playlist:
import numpy as np
#Say f = 2 * x
#So lets give an example of what the input and output should look like
#So the weights can learn from these examples

x= np.array([1,2,3,4],dtype=np.float32)
y= np.array([2,4,6,8],dtype=np.float32) #This is going to be our target

w= 0.0

#model prediction
def forward(x):
    return w*x
#loss
def loss(y, y_predicted):
    return((y_predicted-y)**2).mean() #MSE
#gradients
# MSE = 1/N * (w*x-y)**2 
#dMSE/dw = 1/N 2x (w*x - y)

def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training :f(5) = {forward(5):.3f}', )

#Training:
learning_rate= 0.01
num_of_iterations = 10 

for epoch in range(num_of_iterations):
    #prediction = forwards pass
    y_pred = forward(x)
    #loss
    l= loss(y,y_pred)
    #gradient
    dw= gradient(x,y,y_pred)
    #update weights
    w-=learning_rate*dw

    if epoch%1 == 0: #usually its every other n steps, but in this case they're just ten so we'll get all of them hence the %1 
        print(f'epoch {epoch+1}: w= {w:.3f}, loss={l:.8f}')

print(f'Prediction after training {forward(5):.3f}')

#We can notice that the weights are increasing and the loss is decreasing
