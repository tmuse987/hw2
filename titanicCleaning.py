import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Remove all NA's from Numerics
def removeNAFromNumerics(dfTrain):
    cols = dfTrain.columns
    dfTrainNumeric = dfTrain.select_dtypes(include='number')
    dfTrainNumeric = dfTrainNumeric.drop("PassengerId", axis=1)
    imputer = Imputer(strategy="median")
    imputer.fit(dfTrainNumeric)
    dfTrainNumeric = pd.DataFrame(imputer.transform(dfTrainNumeric), columns = dfTrainNumeric.columns)
    dfTrain = dfTrain.combine_first(dfTrainNumeric)
    dfTrain = dfTrain[cols]
    return dfTrain
    



# Work with non numeric fields
# 

def encodeNonNumeric(dfTrain):
    
    embarked = dfTrain.Embarked
    #change fields from categorical to numerical
    embarkedEncoded, embarkedCategories = embarked.factorize()
    #reshape array
    embarked = embarkedEncoded.reshape(-1,1)
    #use most freqent as this is not really numeric values but transformed values, and -1 is NAN in transform
    imputer = Imputer(strategy="most_frequent", missing_values=-1)
    imputer.fit(embarked)
    embarked = imputer.transform(embarked)
    
    #encode into sparse array the numerical categorical values
    encoder = OneHotEncoder()
    embarked1Hot = encoder.fit_transform(embarked)
    #change to dense array not sparse array so we can reload back to dataframe
    embarkedArray = embarked1Hot.toarray()
    
    #add 3 new columns with 1/0's for all classes of embarked
    dfTrain = dfTrain.assign(Embarked_S = embarkedArray[:,0],Embarked_C = embarkedArray[:,1],Embarked_Q = embarkedArray[:,2])
    dfTrain = dfTrain.drop("Embarked", axis =1)
    return dfTrain



# clean cabin
# Note, we are not replacing NaN's by an imputed value, we are giving it an "unknown" 
# as those NaN's are most likely indicative of a not first class passenger, rather than an unknown cabin
# We also will change the cabin class from categorical to qualitative, as A is 
# indeed best level for cabin and U is indeed worst, we will change T and U to be 
#one and two greater than the highest numerical value we compute for other cabins
def updateCabinColumn(dfTrain):
# remove nan, make those I's (which would be lowest deck level) and then take first letter
# which indicates cabin location deck
    dfTrain.Cabin = [cabin[0] for cabin in dfTrain.Cabin.replace(np.nan, 'I')]
    dfTrain.loc[dfTrain.Cabin == 'T', 'Cabin'] = 'H'
    #convert to numbers from 1 to n
    dfTrain.Cabin = [ord(letter)-64 for letter in dfTrain.Cabin]
    return dfTrain




# Convert male/female to integers 0, 1
def convertSexToNumeric(dfTrain):
    dfTrain.loc[dfTrain.Sex == 'male', 'Sex'] = 0
    dfTrain.loc[dfTrain.Sex == 'female', 'Sex'] = 1
    return dfTrain


# For now, lets drop all non numeric values, name is not likely to influence results 
#(we presume) although possibly title as part of name could (e.g., maybe miss or mrs, 
#is a predictor, or rev vs mr is a predictor) 
# 
# Ticket seems unlikely, with our knowledge of its meaning) to be a useful predictor, 
# i.e., we don't know why some are prefixed with characters, some not, some much 
# larger numbers than others, etc...it may have predictive value, but with each one 
# unique and not understanding schema of data it will probably be hard to use as a predictor

def removeNonCategoricalTextColumns(dfTrain):
    dfTrain = dfTrain.drop(["Name", "Ticket"], axis = 1)
    
    if 'Survived' in dfTrain:
        dfSurvived = dfTrain[['PassengerId', 'Survived']]
    else:
        dfSurvived = pd.DataFrame(dfTrain['PassengerId'])
                        
    return dfTrain, dfSurvived


# Convert remaining columns to numeric since they may no longer be from our various transformations
def convertColsToNumeric(dfTrain):
    dfTrain = dfTrain.apply(pd.to_numeric)
    return dfTrain



# Standardize values to between 0 and 1
def standardizeValues(dfTrain, dfSurvived):
    scaler = MinMaxScaler()
    if 'Survived' in dfSurvived:
        scaledDf = scaler.fit_transform(dfTrain.drop(['PassengerId', 'Survived'], axis = 1))
        scaledDf = pd.DataFrame(scaledDf, columns = dfTrain.drop(["PassengerId", 'Survived'], axis =1).columns)        
        scaledDf['Survived'] = dfSurvived['Survived']
    else:
        scaledDf = scaler.fit_transform(dfTrain.drop('PassengerId', axis = 1))
        scaledDf = pd.DataFrame(scaledDf, columns = dfTrain.drop("PassengerId", axis =1).columns)                
    scaledDf['PassengerId'] = dfSurvived['PassengerId']    
    return scaledDf






