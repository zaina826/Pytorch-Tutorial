import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler #To scale our features
from sklearn.model_selection import train_test_split

# 0) Prep the Data
bc = datasets.load_breast_cancer() #Binary classification Problem
X,y = bc.data, bc.target
 
n_samples, n_features= X.shape
#If we print these, we learn there are 569 samples and 30 features.

#In the previous examples, our functions were easy enough for us 
#To just check our progress by testing f(5) = 10 or something
#In this case, since we have more data, we'll need to keep track 
#of some stats like accuracy for example.
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1234)
#So now, we have X and Y which we'll use for training and X and Y which we'll use for testing.

#Scale:
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
#Since we've already found the scaling parameters: mean, standard deviation, we don't need to find them again
#so we just say transform.
X_test = sc.transform(X_test)
#This means that if we have features that are let's say, between -243 and 395, we'll be able 
#to fit them all between -1 and 1, this ensures consistency, performance, and optimization.

X_train= torch.from_numpy(X_train.astype(np.float32))
X_test= torch.from_numpy(X_test.astype(np.float32))
y_train= torch.from_numpy(y_train.astype(np.float32))
y_test= torch.from_numpy(y_test.astype(np.float32))

#Reshape y tensors, like last tutorial
y_train= y_train.view(y_train.shape[0],1) #So now it will go from one row to one column
y_test= y_test.view(y_test.shape[0],1) 


# 1) Model
# f = wx+b, and sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_features,1)
    def forward(self,x):
        y_predicted= torch.sigmoid(self.linear(x)) #Sigmoid after the linear function
        return(y_predicted)
model = LogisticRegression(n_features)   

# 2) Loss
criterion = nn.BCELoss() #Binary Cross Entropy 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Optimizer
num_of_epochs= 1000
for epoch in range(num_of_epochs):
    y_predicted= model(X_train)
    loss = criterion(y_predicted, y_train) #The training samples and the predicted ys

    loss.backward()

    #Update the weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f'epoch: {epoch+1}, loss= {loss.item():4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_classes = y_predicted.round()
    accuracy = y_predicted_classes.eq(y_test).sum()/float(y_test.shape[0]) #Sum all correcrt predictions then divide them by all the samples
    print(f'accuracy={accuracy:.4f}')
