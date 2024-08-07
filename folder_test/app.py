#https://mms.udemy.com/course/pytorch-for-deep-learning-with-python-bootcamp/learn/lecture/14837954#overview#
import sklearn
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

X = torch.linspace(1,50,50).reshape(-1,1)
torch.manual_seed(71)
e = torch.randint(-8,9,(50,1), dtype=torch.float) #error term, de -8 a 9, na mesma shape do X, com type float
y = 2*X + 1 + e
#now we want to plot this curve. Cant plot a tensor, so need to convert to numpy array
plt.scatter(X.numpy(),y.numpy())
plt.show()
torch.manual_seed(59)
model = nn.Linear(in_features=1, out_features=1) #1 in_feature porque soh tem x, um out_feature, que eh o y
print(model.weight)
print(model.bias)

#set up a class variable called "Model"
class Model(nn.Module): #This line defines a new class Model which is a subclass of nn.Module. nn.Module is a base class for all neural network modules in PyTorch. Your new class will inherit all the functionalities of the nn.Module class.

    def __init__(self,in_features,out_features): #This line defines the initialization method for the class Model. When you create an instance of Model, this method will be called. It takes three parameters: self, in_features, and out_features. self refers to the instance itself and is automatically passed in when you call the method on an instance. in_features is the number of features (or the size of the input layer), and out_features is the number of output features (or the size of the output layer).
        super().__init__() #Inside the __init__ method, this line calls the initialization method of the parent class nn.Module. This is necessary to properly initialize the new Model instance as a PyTorch module.
        self.linear = nn.Linear(in_features,out_features) #"Linear", here, means linear layer, not a linear model
    def forward (self,x): #This method defines the forward pass of the model. In PyTorch, you override the forward method from nn.Module to define the forward pass of your model. The x parameter is the input data.
        y_pred = self.linear(x) #Inside the forward method, this line passes the input x through the linear layer (defined in the __init__ method) and assigns the output to y_pred. The self.linear(x) applies the linear transformation defined by nn.Linear that you initialized earlier.
        return y_pred   
torch.manual_seed(59)

model = Model(1,1)
print(model.linear.weight)
print(model.linear.bias)
#having weight and bias, lets plug x = 2 to see what result we get. We should get (f(x) = 0.1060*2 + 0.9638 = 1.1758)
x = torch.tensor([2.0])
print(model.forward(x))
#since weight and bias, for now, are purely random numbers, we should expect our model to perform very poorly compared to the initial
#numbers and the scatter plot we plotted above. Lets visualize this current bad performance:
x1 = np.linspace(0.0, 50.0, 50)
w1 = 0.1059
b1 = 0.9637
y1 = w1*x1 + b1
plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1)
plt.show()
#in order to make the model learn/optimize, need to set up a loss function. We want the model to reach a weight as close as possible to 2 and a bias as close to 1 as possible, as we defined them above
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001) #Stochastic Gradient Descent as optimizer for the learning rate
epochs = 50 #one epoch is defined as a single path through the entire dataset
losses = [] #a placeholder where we want to keep track of our mean squared errors as we go along
#now we define a for loop where we will keep track of our losses, weights and biases along the epochs

for i in range (epochs):
    i = i + 1

    y_pred = model.forward(X) #calculate prediction for y
    loss = criterion(y_pred,y) #compute the difference between predicted y and actual y
    losses.append(loss) #record the difference in our loss placeholder

    print(f"epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()}) bias: {model.linear.bias.item()}")
    
    optimizer.zero_grad()
          
    loss.backward() #performs the backpropagation
    optimizer.step() #optimizes the hyperparameters as we learn

#now lets use the model, now that we have good parameters for them and MSE converged
x = np.linspace(0.0, 50.0, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
predicted_y = current_weight*x + current_bias
#now let us visualize how the model effectively learned
plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y)
plt.show()

#quick detour for a pandas + pytorch dataframe manipulation
df = pd.read_csv("iris.csv")
df['variety'] = df['variety'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
df = df.rename(columns={'variety': 'target'})
df.head()

from torch.utils.data import TensorDataset , DataLoader
data = df.drop('target', axis=1).values
labels = df['target'].values
#independent variables and dependent variable split
iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))

iris_loader = DataLoader(iris, batch_size=50, shuffle=True) #this will create train sets as batches. Very useful for big datasets
for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)

#Parei em 43 Basic Pytorch ANN - Part One