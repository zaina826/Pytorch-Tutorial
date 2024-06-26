#In this course, we'll be able to replace loss calculation and parameter updates.
#And replace the prediction.

#Design model (input and output and forward pass)
#Construct loss and optimizer
#Training loop: forward pass, backward pass, get gradient, update weights


import torch
import torch.nn as nn #The neural network module

#We have to change the shape: 4 samples, and 1 sample for each feature
x= torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y= torch.tensor([[2],[4],[6],[8]],dtype=torch.float32) #This is going to be our target
# w= torch.tensor(0.0,dtype= torch.float32, requires_grad=True) #This is the weight, it will learn and require gradient 

#Replace weight:
n_samples, n_features= x.shape
print(n_samples, n_features)

input_size= n_features
output_size = n_features
x_test= torch.tensor([5], dtype=torch.float32)

learning_rate= 0.01

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__() 
        #define the layers:
        self.lin = nn.Linear(input_dim, output_dim) #This defines just one layer
    def forward(self,x):
        return self.lin(x)
    #Linear regression basically just does the forward pass
    #y=w⋅x+b

model = LinearRegression(input_size, output_size)
#OR
#model=nn.Linear(input_size, output_size ) #In this case we're also just initializing one layer

#In both scenarios since input and output size is 1, ie we are mapping one input to 
#one output, so we have on weight and one bias.

#I thought this was supposed to be 16 but it's not dense 1 should be connected to 2 only
#2 should be connected to 4 only and so on.
#1 should not be connected to 4,6, or 8.

loss= nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(), learning_rate) #Stochastic gradient descent


print(f'Prediction before training :f(5) = {model(x_test).item():.3f}', )


#Back propagation is not as exact, so we should increase iterations
num_of_iterations = 2600

# Or instead :
# model=nn.Linear(input_size, output_size )

for epoch in range(num_of_iterations):
    #prediction = forwards pass
    y_pred = model(x)
    #loss
    l= loss(y,y_pred)
    #gradient = backward pass in pytorch
    l.backward() #So now this is dw
    #update weights
    optimizer.step()

    #Zero the gradients
    optimizer.zero_grad() #Because usually people accumulate their gradients over mini-batches
    #Pytorch has it built it to accumalate all gradient
    #But in this case, we don't want to accumilate them.
    
    if epoch%10 == 0: #usually its every other n steps, but in this case they're just ten so we'll get all of them hence the %1 
        [w,b]=model.parameters() #Unpacks weights and biases
        print(f'epoch {epoch+1}: w= {w[0][0].item():.3f}, loss={l:.8f}')
        #We actually also have one bias, but it's very small, we can print it.

print(f'Prediction after training :f(5) = {model(x_test).item():.3f}', )

#We can notice that the weights are increasing and the loss is decreasing
