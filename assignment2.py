# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:26:44 2019

@author: Joe Pearson
references:
    Results of pandas.describe function have been manually coded (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
    
"""
# Import relevant files and libraries
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

dataset = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv')
dataset.columns = dataset.columns.str.strip().str.replace(' ', '')

# Get column data
status = dataset['Status'].values
power_sensor1 = dataset['Power_range_sensor_1'].values
power_sensor2 = dataset['Power_range_sensor_2']
power_sensor3 = dataset['Power_range_sensor_3']
power_sensor4 = dataset['Power_range_sensor_4']
pressure_sensor1 = dataset['Pressure_sensor_1']
pressure_sensor2 = dataset['Pressure_sensor_2']
pressure_sensor3 = dataset['Pressure_sensor_3']
pressure_sensor4 = dataset['Pressure_sensor_4']
vibration_sensor1 = dataset['Vibration_sensor_1']
vibration_sensor2 = dataset['Vibration_sensor_2']
vibration_sensor3 = dataset['Vibration_sensor_3']
vibration_sensor4 = dataset['Vibration_sensor_4']


# Function for describing a dataset all to sixth decimal place - similar to how pandas description performs
#def describe(dataset):
#    count = round(len(dataset), 6)
#    mean = round(np.sum(dataset)/len(dataset), 6)
#    var = round(sum(pow(x - mean,6) for x in dataset) / len(dataset),6)
#    std = round(math.sqrt(var),6)
#    minimum = dataset.min()
#    maximum = dataset.max()
#    twentyFivePercent = round(dataset.quantile(.25),6) 
#    fiftyPercent = round(dataset.quantile(.5),6)
#    seventyFivePercent = round(dataset.quantile(.75),6)
#    dtype = dataset.dtype
#    
#    return "Count: " + str(count), "Mean: " + str(mean), "Variance: " + str(var), "Standard Deviation: " + str(std), "Minimum: " + str(minimum), "Maxmimum: " + str(maximum), "25%: " + str(twentyFivePercent), "50%: " + str(fiftyPercent), "75%: " + str(seventyFivePercent), "Data Type: " + str(dtype)

# Loop the data retrieved from the describe function
#for i in describe(pressure_sensor1):
#    print(i)

# Boxplot for the status and vibration sensor classes
#sns.boxplot(y=status, x=vibration_sensor1, width=0.5)
#plt.title('Boxplot for the status of vibration sensor 1')
#plt.xlabel('Vibration Sensor 1')
#plt.show()

# Density plot for the vibration_sensor_2 class
#sns.distplot(vibration_sensor2, hist=True)
#plt.title('Density Plot for Vibration Sensor 2')
#plt.ylabel('Probability Density')
#plt.xlabel('Vibration Sensor 2')
#plt.show()


#f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Add a graph in each part
#sns.boxplot(y=status, x=vibration_sensor1, ax=ax_box)
#sns.distplot(vibration_sensor2, hist=True, ax=ax_hist)
#plt.grid()
# Remove x axis name for the boxplot
#ax_box.set(xlabel='Vibration Sensor 1')
#ax_hist.set(xlabel='Vibration Sensor 2')
#ax_hist.set(ylabel='Probability Densisty Function')
    
# Begin the manual label encoding of the categorical status variable
clean_up = {
            'Status': {'Abnormal': 1, 'Normal': 0 }
        } 
dataset.replace(clean_up, inplace=True)

# Collect the cols that are required (x being all but the status whilst y being the status col)
x = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, [0]].values

# Train - Test Split (Randomised)
def train_test_split(X,Y,train_split):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for position in range(len(X)):
        if position >= len(X)*train_split:
#            
            X_test.append(X[position])
            Y_test.append(Y[position])
        else:
            X_train.append(X[position])
            Y_train.append(Y[position])
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = train_test_split(x,y,0.9)

# ANN architecture creation
# Takes in a specified input and hidden size, within it uses the sigmoid function as the non-linear
# activation function as well as the logistic function
# i.e. 12 -> 500 -> 500 -> 1
class Model(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Model, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.sigmoid = torch.nn.Sigmoid()
            self.fc3 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            sigmoid1 = self.sigmoid(hidden)
            hidden2 = self.fc2(sigmoid1)
            sigmoid2 = self.sigmoid(hidden2)
            output = self.fc3(sigmoid2)
            output = self.sigmoid(output)
            return output
# Creation of tensor objects as floats - variables taken from the train/test split above
x_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(Y_train)
x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(Y_test)

# ANN instantiation
model = Model(x_train.shape[1], 500)
#print(model) # Print the model architecture

# define the loss function using the BCELoss (Binary Cross Entropy) and
# define the optimiser with SGD (Stochastic Gradient Descent), with a learning rate of 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Create variables to be used in the loop below, i.e. the epochs are the number of iteration
epoch = 10
correct = 0
total = 0
accuracyList = []
epochList = []
for epoch in range(epoch):  
    # We first pass the training data to the model before perform a forward pass. After this we compute the loss
    y_pred = model(x_train)
    optimizer.zero_grad()
    loss = criterion(y_pred.view(-1,897), y_test.view(-1,99))
    # Now we perform a backward pass and update weights.
    loss.backward()
    optimizer.step()
    # Below is for calculating the accuracy percentage
    total += y_train.shape[0]
    y_pred = (y_pred>0.5).float()
    correct += (y_pred == y_test).sum().item()
    
    accuracy = 100 * correct / total
#    print("Epoch {} Accuracy: {}".format(epoch, accuracy)) # List accuracy for each epoch
    accuracyList.append(accuracy)
    epochList.append(epoch)
# Print the accuracy of the ANN and plot
accuracy = 100 * correct / total
print("Overal Accuracy: {}%".format(accuracy))
plt.plot(epochList, accuracyList)