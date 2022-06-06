# -------------------
import pandas as pd
import sklearn.impute

air_quality_full = pd.read_excel('./AirQualityUCI.xlsx', usecols=range(2, 15))
# -----------------------
#print( air_quality_full.sample(5))
# -------------------
air_quality = air_quality_full.loc[air_quality_full.iloc[:, 0] != -200, :]
# --------------------------------------------
import numpy as np

ndata, ncols = np.shape(air_quality)  # number of data observations and number of columns in the dataframe
pmissing = np.empty(ncols)  # An empty vector that will keep the percentage of missing values per feature
for i in range(ncols):
    pmissing[i] = (air_quality.iloc[:,i] == -200).sum() / ndata  # Computes the percentage of missing values per column
air_quality = air_quality.loc[:, pmissing < 0.2]

ndata, ncols = np.shape(air_quality)  # number of data observations and number of columns in the dataframe
print(ndata)
print(ncols)
# ---------------------------------------------------------------------
np.random.seed(71480)  # Make sure you use the last five digits of your student UCard as your seed
index = np.random.permutation(ndata)  # We permute the indexes
N = np.int64(np.round(0.70 * ndata))  # We compute N, the number of training instances
Nval = np.int64(np.round(0.15 * ndata))  # We compute Nval, the number of validation instances
Ntest = ndata - N - Nval  # We compute Ntest, the number of test instances
data_training_unproc = air_quality.iloc[index[0:N], :].copy()  # Select the training data
data_val_unproc = air_quality.iloc[index[N:N + Nval], :].copy()  # Select the validation data
data_test_unproc = air_quality.iloc[index[N + Nval:ndata], :].copy()  # Select the test data


import numpy as np
import sklearn as sk
from pandas import DataFrame
from numpy import ndarray



#2.a
from sklearn.impute import SimpleImputer
pip1= SimpleImputer(missing_values=-200, strategy='mean')

from sklearn.preprocessing import StandardScaler
pip2=StandardScaler()

from sklearn.pipeline import Pipeline
pip=Pipeline([('1',pip1),('2',pip2)])


#2.b

def split_x_y(data:DataFrame):
    colNames=list(data)
    y=data[colNames[0]].values
    y=y.reshape(len(y),1)
    x=data.drop(columns=colNames[0]).values
    return x,y
xTrain_unproc,yTrain=split_x_y(data_training_unproc)
xVal_unproc,yVal=split_x_y(data_val_unproc)

pip.fit(xTrain_unproc)
xVal=pip.transform(xVal_unproc)


# 2.c
# step1
from sklearn.model_selection import PredefinedSplit
test_fold = np.hstack((np.ones(N)*-1,np.zeros(Nval)))
ps = PredefinedSplit(test_fold)

# step2
estimators=np.linspace(10,100,5,dtype=int)
features=np.linspace(6,11,5,dtype=int)
samples=np.linspace(200,1000,7,dtype=int)

# step3
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = dict(n_estimators = estimators,max_features = features,max_samples=samples)

forest=RandomForestRegressor()

data_training_Val_unproc = air_quality.iloc[index[0:N+Nval], :].copy()
xTrain_Val_unproc,yTrain_Val=split_x_y(data_training_Val_unproc)

xTrain_Val=pip.fit_transform(xTrain_Val_unproc)


GS=GridSearchCV(forest,param_grid,scoring='r2',cv=ps,error_score=0.0)


GS.fit(xTrain_Val,yTrain_Val.ravel())
# # step4
print(GS.best_score_)
print(GS.best_params_)
print(type(GS.best_params_))

# 2.d
def MSE(y_result,y):
    y_delta = y - y_result
    y_delta2 = np.matmul(y_delta.T, y_delta) / len(y)
    return y_delta2[0][0]

def RMSE(y_result,y):
    mse=MSE(y_result,y)
    return np.sqrt(mse)
def R2(y_result,y):
    mse=MSE(y_result,y)
    y_var=np.var(y)
    re=1-mse/y_var
    return re

xTest_unproc,yTest=split_x_y(data_test_unproc)
xTest=pip.transform(xTest_unproc)
forest=RandomForestRegressor\
    (n_estimators=GS.best_params_['n_estimators']
     ,max_features=GS.best_params_['max_features']
     ,max_samples=GS.best_params_['max_samples'])
forest=RandomForestRegressor(n_estimators=10,max_features=7,max_samples=100)

forest.fit(xTrain_Val,yTrain_Val.ravel())
yTest_result= forest.predict(xTest)
print(yTest_result-yTest)
print(RMSE(yTest_result,yTest))

print(R2(yTest_result,yTest))

# yy=forest.predict(xTrain_Val)
# print(yy-yTrain_Val)
# print(R2(yy,yTrain_Val))
