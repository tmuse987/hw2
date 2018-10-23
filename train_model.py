#!/usr/bin/env python
# coding: utf-8

# In[137]:


import titanicCleaning as cleaner
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#path = "L:/school/cuny/622/cuny622/hw2/"
path = ""

dfTrain = pd.read_csv(path + "titanic.train.csv")
dfTrain = cleaner.removeNAFromNumerics(dfTrain)
dfTrain = cleaner.encodeNonNumeric(dfTrain)
dfTrain = cleaner.updateCabinColumn(dfTrain)
dfTrain = cleaner.convertSexToNumeric(dfTrain)
dfTrain, dfSurvived = cleaner.removeNonCategoricalTextColumns(dfTrain)
dfTrain = cleaner.convertColsToNumeric(dfTrain)
dfTrain = cleaner.standardizeValues(dfTrain, dfSurvived)


# Below, now that we have done the more difficult work of cleaning up and standardizing the data, 
# is where we actually do the work of training the model. The "bagging" (aka bootstrap aggregating) 
# method was chosen, in addition to using a Decision Tree classifier.  Bagging samples 
# a subset of the data a number of different times, and melds the results into one model.  
#It is an improvement as it reduces randomness of a classifier run just once.
# 
# Decision tree was chosen obviously as it is needed to make the bagging classifier into a random forest classifier.

# In[138]:


#divide up (randomly) the dfTrain to see how we are doing
from sklearn.model_selection import train_test_split
xTrain ,xTest = train_test_split(dfTrain,test_size=0.6, random_state = 42)  

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
baggingCl = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, max_samples=100, bootstrap=True, random_state = 42)
baggingCl.fit(xTrain.drop(['PassengerId','Survived'],axis= 1), xTrain['Survived'])

yPred = baggingCl.predict(xTest.drop(['PassengerId','Survived'],axis= 1))
xTestResults = xTest.copy()
xTestResults['yPred'] = yPred
len(xTestResults[(xTestResults['Survived'] == xTestResults['yPred'])])/float(len(xTestResults))


# Save the classifier model to a "pickle" file, which can be used in the next python module of this 
# process (or by anyone else).

# In[139]:


s = pickle.dumps(baggingCl)
with open(path + "classifier.pkl", 'w') as f:
    f.write(s)


# Here we are printing the classification report based on the testing split of our data. 
# (Note the HW problem suggests doing this in the score module, but since we don't know who did
# or did not survive on the test file from kaggle, how can we generate a classfication report from that?)

# In[140]:


from sklearn.metrics import classification_report
print(classification_report(xTestResults['Survived'], xTestResults['yPred']))


# In[136]:


xTrain.drop(['PassengerId','Survived'],axis= 1)

