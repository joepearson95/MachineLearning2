# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:26:44 2019

@author: Joe Pearson
references:
    Results of pandas.describe function have been manually coded (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
    Train/test split adapted from previous assignment and helped by https://www.geeksforgeeks.org/python-random-sample-function/
    
"""
# Import relevant files and libraries
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import math
import random
from pprint import pprint

dataset = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv')
dataset.columns = dataset.columns.str.strip().str.replace(' ', '')

# Function for describing a dataset all to sixth decimal place - similar to how pandas description performs
def describe(dataset):
    count = round(len(dataset), 6)
    mean = round(np.sum(dataset)/len(dataset), 6)
    var = round(sum(pow(x - mean,6) for x in dataset) / len(dataset),6)
    std = round(math.sqrt(var),6)
    minimum = dataset.min()
    maximum = dataset.max()
    twentyFivePercent = round(dataset.quantile(.25),6) 
    fiftyPercent = round(dataset.quantile(.5),6)
    seventyFivePercent = round(dataset.quantile(.75),6)
    dtype = dataset.dtype
    
    return "Count: " + str(count), "Mean: " + str(mean), "Variance: " + str(var), "Standard Deviation: " + str(std), "Minimum: " + str(minimum), "Maxmimum: " + str(maximum), "25%: " + str(twentyFivePercent), "50%: " + str(fiftyPercent), "75%: " + str(seventyFivePercent), "Data Type: " + str(dtype)

# Loop the data retrieved from the describe function
for i in describe(dataset['Pressure_sensor_1']):
   print(i)

# Boxplot for the status and vibration sensor classes
sns.boxplot(y=dataset['Status'], x=dataset['Vibration_sensor_1'], width=0.5)
plt.title('Boxplot for the status of vibration sensor 1')
plt.xlabel('Vibration Sensor 1')
plt.show()

# Density plot for the vibration_sensor_2 class
sns.distplot(dataset['Vibration_sensor_2'], hist=True)
plt.title('Density Plot for Vibration Sensor 2')
plt.ylabel('Probability Density')
plt.xlabel('Vibration Sensor 2')
plt.show()

# Create the two graphs together
f, (combineBox, combineHist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Add graphs together and display
sns.boxplot(y=dataset['Status'], x=dataset['Vibration_sensor_1'], ax=combineBox)
sns.distplot(dataset['Vibration_sensor_2'], hist=True, ax=combineHist)
plt.grid()
combineBox.set(xlabel='Vibration Sensor 1')
combineHist.set(xlabel='Vibration Sensor 2')
combineHist.set(ylabel='Probability Densisty Function')
plt.show()

# Begin the manual label encoding of the categorical status variable if the ANN is being called
clean_up = {
    'Status': {'Abnormal': 1, 'Normal': 0 }
} 
dataset.replace(clean_up, inplace=True)

# Artificial Neural Network

# Collect a random train test split 
def train_test_split(df, test_size):
    # check that the required size given is a float and obtain the test size percentage from param
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    # access the index column and select random values from the dataset based on this
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    # return the relevant train and test datasets
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

# Check for occurances of classifiers within dataset
def check_purity(data):
    # retrieve the Status column
    label_column = data[:, 0]
    unique_classes = np.unique(label_column) # count the occurances of classifications

    # If only one type occurs, return True. Otherwise, return false
    if len(unique_classes) == 1:
        return True
    else:
        return False
    
# Classifies the dataset given
def classify_data(data):
    # retrieve the first column - the Status (abnormal/normal) column
    label_column = data[:, 0]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True) # Retrieve the two classifications and count them
    
    #retrieve the classification with the most occurances (i.e abnormal - 443, normal 453)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index] 
    return classification

# Retrieve the potential splits that the decision tree may make
def get_potential_splits(data, rand_space):
    potential_splits = {} # dictionairy created for obtaining the potential splits in the foor loops below
    _, no_cols = data.shape # Get the shape of a given dataset
    # Loop over the dataset given minus the Status column
    indices = list(range(1, no_cols))
    
    # Retrieve a random amount of indices from a given dataset based on the amount supplied in the rand_space param
    if rand_space and rand_space <= len(indices):
        indices = random.sample(population=indices, k=rand_space)
    
    for col_index in indices:
        values = data[:, col_index] # get particular columns values
        unique_vals = np.unique(values) # obtain unique values
        # If the given feature is not categorical then loop over the unique values and find the potential split
        # otherwise, it is categorical data and compute accordingly
        ft_type = FEAT_TYPES[col_index]
        if ft_type == "continuous":
            potential_splits[col_index] = []
            # Loop over the amount of unique values available
            for indx in range(len(unique_vals)):
                # Ignore the first value due to it not having a previous value to iterate over
                if indx != 0:
                    # Obtain the current/previous values and find the potential split of said values
                    current_val = unique_vals[indx]
                    prev_val = unique_vals[indx -1]
                    potential_split = (current_val + prev_val) / 2
                    potential_splits[col_index].append(potential_split) # append to the dictionairy at the beginning of the function
        else:
            potential_splits[col_index] = unique_vals # add the unique values to the dictionairy 
    return potential_splits

def split_data(data, split_col, split_val):
    split_col_vals = data[:, split_col] # Retrieve all values from the split_col param
    
    # Check the feature supplied from the split_col param. If it is continuous, treat it as a numerical value accordingly
    # otherwise, treat it as the str categorical value it is
    ft_type = FEAT_TYPES[split_col]
    if ft_type == "continuous":
        below = data[split_col_vals <= split_val]
        above = data[split_col_vals > split_val]
    else:
        below = data[split_col_vals == split_val]
        above = data[split_col_vals != split_val]
    return above, below

# Calculates the entropy of a given dataset
def entropy(data):
    status_col = data[:, 0] # retrieve the Status column
    _, counts = np.unique(status_col, return_counts=True) # Count the unique values within this
    probability_status = counts / counts.sum() # Calculate the probability of said column

    # Calculate the entropy using its forumla before returning it
    entropy_formula = sum((probability_status * -np.log2(probability_status)))
    return entropy_formula

# Calculate the total entropy, taking in the split above/below data
def total_entropy(below, above):
    # Retrieve the len of both of these added together and points of said section also
    n_dp = len(below) + len(above)
    p_below = len(below) / n_dp
    p_above = len(above) / n_dp
    # Calculate the total entropy using its formula before returning it
    calc_total_entropy = (p_below * entropy(below) 
                         +p_above * entropy(above))
    return calc_total_entropy

# Function to loop over all potential splits within the dataset given, calculate the total entrophy for said split
# and if this current total entropy of said split is lower than the abritary split number given then we return these two values
def determine_bs(data, ps):
    arb_total_entropy = 999 # Arbritary total entropy value 
    # Loop over said splits keys, and then loop over the elements in list
    for col_indx in ps:
        for val in ps[col_indx]:
            # Split a dataset based on the given column index and value. Returning the above/below splits
            d_above, d_below = split_data(data, split_col=col_indx, split_val=val)
            # Calculate the current total entropy of current loop
            curr_total_entorpy = total_entropy(d_above, d_below)
            # If current split is lower or equal than arbritary total split then assign the current col/val to best split variables that will be returned            
            if curr_total_entorpy <= arb_total_entropy:
                arb_total_entropy = curr_total_entorpy
                bs_col = col_indx
                bs_val = val
    return bs_col, bs_val


def type_of_feat(dataframe):
    types = []
    n_unqiue_thres =  15
    for col in dataframe.columns:
        unique_val = dataframe[col].unique()
        ex_val = unique_val[0]
        
        if(isinstance(ex_val, str)) or (len(unique_val) <= n_unqiue_thres):
            types.append("categorical")
        else:
            types.append("continuous")
    return types

# Creation of decision tree, taking in a dataframe. Keeping it simple as possible for main program
def decision_tree(dataframe, count=0, min_samples=2, max_depth=5, rand_space=None):
    # Checking if it is a np 2-d array. Then instantiating the data variable accordingly
    if count == 0:
        global COL_H, FEAT_TYPES # Create the global variables
        COL_H = dataframe.columns
        FEAT_TYPES = type_of_feat(dataframe)
        data = dataframe.values
    else:
        data = dataframe
        
    # Due to being recursive, checking for purity before beginning the recursion will prevent infinite looping
    if (check_purity(data)) or (len(data) < min_samples) or (count == max_depth):
        classify = classify_data(data)
        return classify
    else:
        count += 1
        
        # Instantiation of the functions above for use in the recursion
        ps = get_potential_splits(data, rand_space)
        split_col, split_val = determine_bs(data, ps)
        above, below = split_data(data, split_col, split_val)
        
        # Sub tree creation taking in the global column header and formating the output accorindgly
        feature_name = COL_H[split_col]
        ft_type = FEAT_TYPES[split_col]
        if ft_type == "continuous":
            question = "{} <= {}".format(feature_name, split_val)
        else:
           question = "{} = {}".format(feature_name, split_val)
        
        sub_tree = {question: []}
        
        # Variable creation for aspects of the decision tree
        yes_ans = decision_tree(below, count, min_samples, max_depth, rand_space)
        no_ans = decision_tree(above, count, min_samples, max_depth, rand_space)
        # Check for end of tree, if at the end then instead of displaying a dictionairy we display the final values
        if yes_ans == no_ans:
            sub_tree = yes_ans
        else:
            # Append the answers to the sub_tree before finally returning it
            sub_tree[question].append(yes_ans)
            sub_tree[question].append(no_ans)
        return sub_tree

# Function to classify a given column, taking in said column and relevant tree
def class_ex(example, tree):
    # Creation of variable holding the current algorithms 'question'. Before splitting said question into its relevant parts
    question = list(tree.keys())[0]
    ft_name, comp_op, val = question.split()
    
    # Begin asking the above 'question'. If yes, get the first element, otherwise get the second element
    if comp_op == "<=":
        if example[ft_name] <= float(val):
            ans = tree[question][0]
        else:
            ans = tree[question][1]
    else:
        if str(example[ft_name]) == val:
            ans = tree[question][0]
        else:
            ans = tree[question][1]
    # If the answer from above is not a dictionairy then return the said classification, otherwise begin recursion aspect
    if not isinstance(ans, dict):
        return ans
    else:
        last_tree = ans
        return class_ex(example, last_tree)
def tree_prediction(test_df, tree):
    prediction = test_df.apply(class_ex, args=(tree,), axis=1)
    return prediction

def accuracy(dataframe, status):
    predictions_correct = dataframe == status
    accuracy = predictions_correct.mean()
    return accuracy

# Function to return a given dataset with randomised indices based on the length of the dataset given and the number of rows wanted
def bootstraper(train_df, n_bootstrap):
    indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap) # Random int in range of dataset len and rows required
    final_bootstrap = train_df.iloc[indices] # The dataset to be returned
    return final_bootstrap

# Function to create a random forest based on the supplied dataset, amount of trees, features, its max depth and given indices
def rand_forest_alg(train_df, trees, n_bootstrap, n_features, max_depth):
    forest = [] # List of the threes
    # Iterate the range of the trees param given, obtaining the indices from bootstraper function based on the
    # dataset given and amount given. Then create a tree based on said data, the depth supplied from the param and
    # the random amount of indices supplied from the features param
    for t in range(trees):
        bootData = bootstraper(train_df, n_bootstrap)
        tree = decision_tree(bootData, max_depth=max_depth, rand_space=n_features)
        forest.append(tree)
    return forest

def forest_prediction(test_df, forest):
    predictions = {} # Dictionairy containing all predictions
    # Loop over the length of a given forest and assign the trees number to a string and then to a variable called col_key
    # then obtain the prediction for the tree based on each loop before appending said prediction to the predcition dictionairy
    for i in range(len(forest)):
        col_key = "tree_{}".format(i)
        tree_predictions = tree_prediction(test_df, tree=forest[i])
        predictions[col_key] = tree_predictions
    
    # Obtain a pandas dataframe with the predictions dictionairy and show it based on a basic index
    predictions = pd.DataFrame(predictions)
    rand_prediction = predictions.mode(axis=1)[0] # the forest prediction based on the mode of the given predictions from before
    return rand_prediction

# ANN architecture creation
# Takes in a specified input and hidden size, within it uses the sigmoid function as the non-linear
# activation function as well as the logistic function
# i.e. 13 -> 500 -> 500 -> 13
class Model(torch.nn.Module):   
        def __init__(self, input_size, hidden_size):
            super(Model, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.sigmoid2 = torch.nn.Sigmoid()
            self.fc3 = torch.nn.Linear(self.hidden_size, 13)
            self.output = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            sigmoid1 = self.sigmoid(hidden)
            hidden2 = self.fc2(sigmoid1)
            sigmoid2 = self.sigmoid(hidden2)
            output = self.fc3(sigmoid2)
            output = self.sigmoid(output)
            return output

# Create variables to be used in the loop below, i.e. the epochs are the number of iteration
def train_model(epoch, train_df, test_df):
    x_train = torch.FloatTensor(train_df.values)
    y_test = torch.FloatTensor(test_df.values)
    # ANN instantiation
    model = Model(x_train.shape[1], 500)
    print(model) # Print the model architecture
    
    # define the loss function using the BCELoss (Binary Cross Entropy) and
    # define the optimiser with SGD (Stochastic Gradient Descent), with a learning rate of 0.01
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1)
    
    y_padData = F.pad(input=y_test, pad=(0,0,0,796), mode='constant', value=0)
    correct = 0
    total = 0
    accuracyList = []
    # Iterate over epochs range
    for epoch in range(epoch):  
        # We first pass the training data to the model before perform a forward pass. After this we compute the loss
        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_padData)
        # Now we perform a backward pass and update weights.
        loss.backward()
        optimizer.step()
        # Below is for calculating the accuracy percentage
        total += x_train.shape[0]#len(x_train)
        y_pred = (y_pred>0.5).float()
        correct += (y_pred == x_train).sum().item()
        accuracy = 100 * correct / total
        # print("Epoch {} Accuracy: {}%".format(epoch + 1, round(accuracy,2))) # List accuracy for each epoch, added 1 to epoch for readability purposes
        accuracyList.append(accuracy)
    return accuracyList

def cross_validation(dataset, k_folds, cv_type, neurons=500, trees=4,):
    k_folded_data = np.array_split(dataset, k_folds)
    cv_accuracy = []
    correct = 0
    total = 0
    # Iterate over epochs range
    for i in range(k_folds):  
        train_df = k_folded_data.copy() # creating a copy of the split dataset for working purposes
        test_df = k_folded_data[i] # select a section of this array based on index supplied in iteration
        del train_df[i] # delete the test set from this iteration before concatenating the remainding sets for training
        train_df = pd.concat(train_df, sort=False)
        if cv_type == "ANN":
            padData = F.pad(input=torch.FloatTensor(test_df.values), pad=(0,0,0,(train_df.shape[0] - test_df.shape[0])), mode='constant', value=0)
            model = Model(torch.FloatTensor(train_df.values).shape[1], neurons)
            # We first pass the training data to the model before perform a forward pass. After this we compute the loss
            y_pred = model(torch.FloatTensor(train_df.values))
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = 1)
            optimizer.zero_grad()
            loss = criterion(y_pred, padData)
            # Now we perform a backward pass and update weights.
            loss.backward()
            optimizer.step()
            # Below is for calculating the accuracy percentage
            total += torch.FloatTensor(train_df.values).shape[0]
            y_pred = (y_pred>0.5).float()
            correct += (y_pred == torch.FloatTensor(train_df.values)).sum().item()
            curr_forest_accuracy = 100 * correct / total
            # print("Epoch {} Accuracy: {}%".format(i + 1, round(curr_forest_accuracy,2))) # List accuracy for each epoch, added 1 to epoch for readability purposes
            cv_accuracy.append(curr_forest_accuracy)
        else:
            forest = rand_forest_alg(train_df, trees=4, n_bootstrap=800, n_features=2, max_depth=4)
            predictions = forest_prediction(test_df, forest)
            curr_accuracy = accuracy(predictions, test_df.Status)
            cv_accuracy.append(curr_accuracy)
    return cv_accuracy

# Get a 90:10 random data split
train_df, test_df = train_test_split(dataset, test_size=0.1)

# Calculate the accuracy of a random tree forest classifier
randforest_accuracy = []
forest = rand_forest_alg(train_df, trees=4, n_bootstrap=800, n_features=2, max_depth=4)
predictions = forest_prediction(test_df, forest)
forest_accuracy = accuracy(predictions, test_df.Status)
randforest_accuracy.append(forest_accuracy)
print("Random Forest accuracy: {}%".format(round(100 * np.mean(randforest_accuracy))))

# Accuracy of the ANN
accuracyList = train_model(10, train_df, test_df)
print("Accuracy of ANN: {}%".format(round(np.average(accuracyList),2)))
k_fold_no = 10
# Print the accuracy of the random forest classifier
forest_cv_accuracy = cross_validation(dataset, k_fold_no, cv_type="Forest")
print("Random Forest accuracy with cross validation: {}%".format(round(100 * np.mean(forest_cv_accuracy))))
# Show Graph
plt.xlabel("K Folds")
plt.ylabel("Accuracy")
plt.title("CV Accuracy of Random Forest")
plt.plot(range(0,k_fold_no), forest_cv_accuracy)
plt.show()
# Print the accuracy of the ANN
ann_cv_accuracy = cross_validation(dataset, k_fold_no, cv_type="ANN", neurons=50)
print("ANN accuracy with cross validation: {}%".format(round(np.average(ann_cv_accuracy))))
# Show Graph
plt.xlabel("K Folds")
plt.ylabel("Accuracy")
plt.title("CV Accuracy of ANN")
plt.plot(range(0,k_fold_no), ann_cv_accuracy)
plt.show()