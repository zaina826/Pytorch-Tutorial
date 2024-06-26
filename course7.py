import torch
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare Data:
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise =20, random_state=1)
X= torch.from_numpy(X_numpy.astype(np.float32))
Y= torch.from_numpy(Y_numpy.astype(np.float32))

Y=Y.view(Y.shape[0],1) 

n_sample, n_features = X.shape

# 1) Model
input_size=n_features
output_size= 1

model = nn.Linear(input_size, output_size)

# 2) Define Loss and Optimizer
criterion= nn.MSELoss()
learing_rate=0.01
optimizer= torch.optim.SGD(model.parameters(), lr=learing_rate)


# 3) Training Loop
num_epochs= 100
for epoch in range(num_epochs):
    #Forward Pass 
    y_predicted= model(X)
    loss=criterion(y_predicted, Y)

    #Back Propagation
    loss.backward()

    #Update the weight and bias:
    optimizer.step()

    #Zero out the gradients 
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy,'ro')
plt.plot(X_numpy, predicted,'b')
plt.show()