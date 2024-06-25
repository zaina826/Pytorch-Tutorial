import torch

x= torch.tensor([1,2,3,4],dtype=torch.float32)
y= torch.tensor([2,4,6,8],dtype=torch.float32) #This is going to be our target
w= torch.tensor(0.0,dtype= torch.float32, requires_grad=True) #This is the weight, it will learn and require gradient 

#model prediction
def forward(x):
    return w*x
#loss
def loss(y, y_predicted):
    return((y_predicted-y)**2).mean() #MSE
#gradients
# MSE = 1/N * (w*x-y)**2 
#dMSE/dw = 1/N 2x (w*x - y)


print(f'Prediction before training :f(5) = {forward(5):.3f}', )

#Training:
learning_rate= 0.01

#Back propagation is not as exact, so we should increase iterations
num_of_iterations = 100

for epoch in range(num_of_iterations):
    #prediction = forwards pass
    y_pred = forward(x)
    #loss
    l= loss(y,y_pred)
    #gradient = backward pass in pytorch
    l.backward() #So now this is dw
    #update weights
    with torch.no_grad(): #We use this to ensure that these operations don't affect the computational graph
        w-=learning_rate*w.grad

    #Zero the gradients
    w.grad.zero_() #Because usually people accumulate their gradients over mini-batches
    #Pytorch has it built it to accumalate all gradient
    #But in this case, we don't want to accumilate them.
    
    if epoch%10 == 0: #usually its every other n steps, but in this case they're just ten so we'll get all of them hence the %1 
        print(f'epoch {epoch+1}: w= {w:.3f}, loss={l:.8f}')

print(f'Prediction after training {forward(5):.3f}')

#We can notice that the weights are increasing and the loss is decreasing
