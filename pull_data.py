#!/usr/bin/env python
# coding: utf-8

# In[25]:


import requests
import pandas as pd
import io

#path = "L:/school/cuny/622/cuny622/hw2/"
path = ""
def pullTitanicFile():
    #code modified from https://stackoverflow.com/questions/50863516/issue-in-extracting-titanic-training-data-from-kaggle-using-jupyter-notebook
    #userNamePW file has format: username,pw
    userNamePW = pd.read_csv(path + 'userNamePW.csv', header=None)

    payload = {
        '__RequestVerificationToken': '',
        'username': userNamePW[0],
        'password': userNamePW[1],
        'rememberme': 'false'
    }

    loginURL = 'https://www.kaggle.com/account/login'
    dataTrainURL = 'https://www.kaggle.com/c/titanic/download/train.csv'
    dataTestURL = 'https://www.kaggle.com/c/titanic/download/test.csv'   

    
    with requests.Session() as c:
        try:
            #login in to kaggle
            response = c.get(loginURL).text
            AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
            print("AntiForgeryToken={}".format(AFToken))
            payload['__RequestVerificationToken']=AFToken
            c.post(loginURL + "?isModal=true&returnUrl=/", data=payload)
        except: 
            print('Error Logging in')
            raise

        #get train dataset
        try:
            response = c.get(dataTrainURL)
            titanicIO =  io.StringIO(response.text)
            dfTitanicTrain =  pd.read_csv(titanicIO)
            filename = path + "titanic.train.csv"
            dfTitanicTrain.to_csv(filename, index=False)
        except:
            print('Error Downloading Train File')
            raise

        #getTestDataset
        try:
            response = c.get(dataTestURL)
            titanicIO =  io.StringIO(response.text)
            dfTitanicTest =  pd.read_csv(titanicIO)
            filename = path + "titanic.test.csv"
            dfTitanicTest.to_csv(filename, index=False)
        except:
            print('Error Downloading Test File')
            raise

        return dfTitanicTrain,dfTitanicTest;  

dfTrain, dfTest = pullTitanicFile()

if (len(dfTrain) < 800 or len(dfTrain.columns) < 11):
    print 'Train file not completely downloaded'
    raise Exception('Train file not completely downloaded')
    
if (len(dfTest) < 400 or len(dfTrain.columns) < 11):    
    print 'Train file not completely downloaded'
    raise Exception('Test file not completely downloaded')

