#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import pickle
import titanicCleaning as cleaner

path = "L:/school/cuny/622/cuny622/hw2/"
path = ""

baggingCl = None
try:
    with open(path + "classifier.pkl", 'r') as f:
        baggingCl = pickle.load(f)
except:
    print 'Error loading "pickle" classifier file, process aborted'
    raise
    
try:    
    dfTest = pd.read_csv(path + "titanic.test.csv")
except:
    print 'Error loading testing csv file, process aborted'
    raise



##run same set of cleaing steps we did for training, using titanicCleaning file
dfTest = cleaner.removeNAFromNumerics(dfTest)
dfTest = cleaner.encodeNonNumeric(dfTest)
dfTest = cleaner.updateCabinColumn(dfTest)
dfTest = cleaner.convertSexToNumeric(dfTest)
dfTest, dfSurvived = cleaner.removeNonCategoricalTextColumns(dfTest)
dfTest = cleaner.convertColsToNumeric(dfTest)
dfTest = cleaner.standardizeValues(dfTest, dfSurvived)
dfTest
    


# create the y predictions
yPred = baggingCl.predict(dfTest.drop('PassengerId',axis= 1))


# create a .csv that can be uploaded to kaggle to get a score
dfResults = pd.DataFrame([dfTest['PassengerId'], yPred]).T
dfResults.to_csv(path + "predictedScore.csv", index = False)


