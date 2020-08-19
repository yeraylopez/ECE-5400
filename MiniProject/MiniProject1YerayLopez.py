#!/usr/bin/env python
# coding: utf-8

# # Mini Project #1: Predicting logic errors of digital circuits

# ## Approach & Reasoning:
# 
# Provided data is presented as an extensionless file  
# File must first be read into python in order to parse it to the appropriate data structure  
# Once data is in the proper format we will analyze it's shape according to:   
# 
# X*theta = Y, where X is our input data and Y is our output
# 
# X our input data will always be 32 bits long  
# 
# Input data shape: 15000,32  
#   
# 
# Y = each of the 32 column 
# 
# Output data label shape: 15000,1
# 
# **Based on this information 32 individual models must be created in order to predict all 32 bits**

# ## Part 1: Preparing Data
# ### Import Libraries

# In[1]:


import numpy as np 
from datetime import datetime
start_time = datetime.now()


# In[2]:


# function converts any file seperated by newlines to a numpy array
# accounts for any whitespace in the file
def file2Array(fileName, datatype):
    fileData = [line.rstrip('\n') for line in open(fileName)]
    
    dataList = [] 
    for i in range(len(fileData)):
        temp = "".join(fileData[i].split())
        dataList.append(temp)
    #print(dataList[0])

    # map() can listify the list of strings individually 
    dataList = list(map(list, dataList)) 
    #print(dataList[0]) 

    dataArray = np.asarray(dataList, dtype=datatype)
    #print(dataArray.shape)
    
    return dataArray


# In[3]:


# prepare numpy arrays for machine learning model
trainingData  = file2Array('training_data' ,int)
trainingLabel = file2Array('training_label',int)
testingData   = file2Array('testing_data'  ,int)
testingLabel  = file2Array('testing_label' ,int) 

# count the number of feature in the given label file
featureCount    = np.size(trainingLabel,1)


# ## Part 2: Build Machine Learning Model using scikit-learn
# ### 4 Step Process [I,M,T,P]
# 1. **Import**
# 2. **Make**
# 3. **Train**
# 4. **Predict**

# ### Example workflow:
# ### Step 1: [I]mport Model
# 
# ```from sklearn import tree ```
# ### Step 2: [M]ake Model
# ```clf = tree.DecisionTreeClassifier()```
# ### Step 3: [T]rain Model
# ``` clf.fit(trainingData,trainingLabel) ```
# ### Step 4: [P]redict with Model
# ```predictions = clf.predict(testingData)```
# ### Gather Metrics
# ```aScore = accuracy_score(testingLabelCol, predictions)
# pScore = precision_score(testingLabelCol, predictions)
# rScore = recall_score(testingLabelCol, predictions)```

# In[4]:


# import Model
from sklearn import tree
# import desired metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# ## Decision Tree 89.999%

# In[5]:


aScoreAvg,pScoreAvg,rScoreAvg = 0,0,0
aScoreTotal,pScoreTotal,rScoreTotal = 0,0,0
# Make Model
clf = tree.DecisionTreeClassifier(max_depth=12)

print("Decision Tree Results: ")

# Train model for all 32 features 

for i in range(featureCount):

    # extract column
    trainingLabelCol = (trainingLabel[:, [i]]).ravel()
    testingLabelCol = (testingLabel[:, [i]]).ravel()
    #print(trainingLabel) #debug
    #print(trainingData)  #debug
    
    if((len(np.unique(trainingLabelCol)) and len(np.unique(testingLabelCol))) == 1):
        aScore = 1.0
        pScore = 1.0
        rScore = 1.0
        
        aScoreTotal += aScore
        pScoreTotal += pScore
        rScoreTotal += rScore
        #print(i,score,np.unique(trainingLabelCol),np.unique(testingLabelCol)) #print unique elements
        print(i,aScore,pScore,rScore)
        continue
    
    # fit model
    clf.fit(trainingData,trainingLabelCol)
    
    # generate all predictions
    predictions = clf.predict(testingData)
    
    # obtain metrics (accuracy, precision, recall)
    aScore = accuracy_score(testingLabelCol, predictions)
    pScore = precision_score(testingLabelCol, predictions, average='weighted') 
    rScore = recall_score(testingLabelCol, predictions, average='weighted')
    
    # Sum all scores to average at the end
    aScoreTotal += aScore
    pScoreTotal += pScore
    rScoreTotal += rScore
    
    #print(i,aScore,pScore,rScore,np.unique(trainingLabelCol),np.unique(testingLabelCol)) #extended print func
    print(i,aScore,pScore,rScore)

# also works also dont forget the rest of the metrics jiao wants
#accuracy_score((testingLabel[:, [4]]).ravel(), predictions)
aScoreAvg = aScoreTotal/featureCount
pScoreAvg = pScoreTotal/featureCount
rScoreAvg = rScoreTotal/featureCount

print("\n")
print("Mean Accuracy: ", aScoreAvg)
print("Mean Precision: ", pScoreAvg)
print("Mean Recall: ", rScoreAvg)


# In[6]:


time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

