# -------------------
import pandas as pd

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


# 1.a
from pandas import DataFrame
from numpy import ndarray

def DealtMissingValue(data:DataFrame):
    row,cow=np.shape(data)
    averages= np.empty(cow)
    stds=np.empty(cow)
    data= data.replace(-200,np.nan)
    for i in range(cow):
        if i==0:continue
        cowi: DataFrame=data.iloc[:,i]
        # get average
        averages[i]=cowi.mean()
        # change nan to average
        data.iloc[:, i]=cowi.fillna(averages[i])
        # get std and Standardize
        stds[i]=data.iloc[:,i].std()
        for j in range(row):
            data.iat[j,i]=(data.iat[j,i]-averages[i])/stds[i]
    return data,averages,stds

data_training_unproc,avers,stds=DealtMissingValue(data_training_unproc)

# print(data_training_unproc.sample(10))
# print(avers)
# print(stds)

# 1.4
colNames=list(data_training_unproc)
yTrain=  data_training_unproc[colNames[0]].values
yTrain=yTrain.reshape(len(yTrain),1)

XTrain = data_training_unproc.drop(columns=colNames[0]).values


# print(yTrain.shape)
# print(XTrain.shape)

# print(type(yTrain))
# print(type(XTrain))

# 1.b
# step1
def standardiseDF(data,averages,stds):
    data= data.replace(-200,np.nan)
    row,cow=np.shape(data)
    for i in range(cow):
        if i==0:continue
        cowi=data.iloc[:,i]
        data.iloc[:, i]=cowi.fillna(averages[i])
        for j in range(row):
            data.iat[j,i]=(data.iat[j,i]-averages[i])/stds[i]
    return data
data_val_unproc=standardiseDF(data_val_unproc,avers,stds)

def split_x_y(data:DataFrame):
    colNames=list(data)
    y=data[colNames[0]].values
    y=y.reshape(len(y),1)
    x=data.drop(columns=colNames[0]).values
    return x,y

XVal,yVal=split_x_y(data_val_unproc)


# print(data_val_unproc.sample(10))

# step2
Paras = np.logspace(-1,1,5,base=2)   #  𝛾 , the regularisation parameter
Lrs = np.logspace(-2.7,-4.7,5,base=10)  #𝜂 , the learning rate
PointCounts = np.linspace(30,300,5) #𝑆 , the number of datapoints
print(Paras)
print(Lrs)
print(PointCounts)


# step3 reclam

# def train(feature:DataFrame,target:DataFrame,ridge_para,lr,times):
#
#     target=target.to_numpy().reshape(len(target),1)
#     feature=addX_0(feature)
#     ww=np.zeros([feature.shape[1],1])
#
#     lastloss=0
#     i=0
#     while i<times:
#         i+=1
#         lr_new=lr
#         whiletimes = 0
#         while True:
#             # xTxW-xTy
#             w_ = np.matmul(np.matmul(feature.T, feature), ww) + ridge_para * ww - np.matmul(feature.T, target)
#
#             w_new = ww - lr_new * w_
#             w_new_loss=getLoss(feature, target, w_new, ridge_para)
#             if i==1 or w_new_loss<lastloss:
#                 print(i,whiletimes, '     loss reduce: ',lastloss-w_new_loss," new loss: ",w_new_loss)
#                 ww=w_new
#                 lastloss=w_new_loss
#                 break
#             whiletimes+=1
#             lr_last=lr_new
#             lr_new/=2
#             # print(lr_new)
#             if lr_last==lr_new:
#                 i=times
#                 break
#     print(w_new_loss)
#     return ww

# def train2(feature:DataFrame,target:DataFrame,ridge_para,lr,times):
#
#     target=target.to_numpy().reshape(len(target),1)
#     feature=addX_0(feature)
#     ww=np.zeros([feature.shape[1],1])
#
#     lastloss=0
#     i=0
#     while i<times:
#         i+=1
#
#         w_ = np.matmul(np.matmul(feature.T, feature), ww) + ridge_para * ww - np.matmul(feature.T, target)
#         ww = ww - lr * w_
#         NewLoss = getLoss(feature, target, ww, ridge_para)
#         # print(i, '     loss reduce: ', lastloss - NewLoss, " new loss: ", NewLoss)
#         lastloss=NewLoss
#
#     print(times,"final loss: ",lastloss)
#     return ww,lastloss

# def train_single(feature:DataFrame,target:DataFrame,ridge_para,lr,times):
#     target=target.to_numpy().reshape(len(target),1)
#     feature=addX_0(feature)
#     ww=np.zeros([feature.shape[1],1])
#
#     lastloss=0
#     i=0
#     while i<times:
#         i+=1
#
#         for one_feature,one_target in zip(feature,target):
#             one_feature=one_feature.reshape(1,len(one_feature))
#             one_target=one_target.reshape(1,1)
#             w_ = np.matmul(np.matmul(one_feature.T, one_feature), ww) + ridge_para * ww - np.matmul(one_feature.T, one_target)
#             ww = ww - lr * w_
#         new_loss = getLoss(feature, target, ww, ridge_para)
#         print(i, '     loss reduce: ', lastloss - new_loss, " new loss: ", new_loss)
#         lastloss = new_loss
#     return ww,lastloss


# step3
def get_loss(x,y,w,ridge_para):

    error=(y-np.matmul(x,w))
    loss=np.matmul(error.T,error)+ridge_para*np.matmul(w.T,w)
    return loss

def add_1s_before_x(feature:ndarray):
    feature_1=np.ones([feature.shape[0],1])
    re=np.concatenate((feature_1,feature),axis=1)
    return re.reshape(feature.shape[0],feature.shape[1]+1)

def MSE(x,y,w):
    y_delta = y - np.matmul(x, w)
    y_delta2 = np.matmul(y_delta.T, y_delta) / len(y)
    return y_delta2[0][0]

def RMSE(x,y,w):
    mse=MSE(x,y,w)
    return np.sqrt(mse)

def R2(x,y,w):
    mse=MSE(x,y,w)
    y_var=np.var(y)
    re=1-mse/y_var
    return re

def train_single_group(feature,target,times,ridge_para,lr,count):
    ww = np.zeros([feature.shape[1]+1, 1])
    return train_single_group_(feature,target,times,ridge_para,lr,count,ww)

def train_single_group_(feature:ndarray,target:ndarray,times,ridge_para,lr,count,ww):
    collection_len=len(target)
    feature= add_1s_before_x(feature)

    lastloss = 0
    i = 0
    while i < times:
        i += 1
        # calculate start_index end_index
        start_index=count*(i-1)
        end_index=count*i
        page1=start_index//collection_len
        page2=end_index//collection_len
        index1 = start_index % collection_len
        index2 = end_index % collection_len
        #prepare datas
        if page1==page2:
            one_target = target[index1:index2]
            one_feature= feature[index1:index2]
        else:

            one_target=np.concatenate((target[index1:collection_len],target[0:index2]))
            one_feature=np.concatenate((feature[index1:collection_len],feature[0:index2]))
        w_ = np.matmul(np.matmul(one_feature.T, one_feature), ww) + ridge_para * ww - np.matmul(one_feature.T,one_target)
        ww = ww - lr * w_


        # new_r2=R2(feature,target,ww)
        # print(i,index1,index2 ,'     r2 increase: ', new_r2 - lastloss, " new r2: ", new_r2)
        # lastloss=new_r2

        # new_loss = get_loss(feature.values, target.values, ww, ridge_para)
        # # print(i,index1,index2 ,'     loss reduce: ', lastloss - new_loss, " new loss: ", new_loss)
        # lastloss = new_loss

    return ww

min_rmse=1000
best_para=0
best_lr=0
best_p_count=0
for i in range(len(Paras)):
    for j in range(len(Lrs)):
        for k in range(len(PointCounts)):
            print(Paras[i],int(PointCounts[k]),"       ",Lrs[j])
            w = train_single_group(XTrain, yTrain, 200, Paras[i], Lrs[j], int(PointCounts[k]))
            rmse=RMSE(add_1s_before_x(XVal), yVal, w)
            print(i,j,k,"                               ", rmse)
            print()
            if rmse<min_rmse:
                min_rmse=rmse
                best_para=Paras[i]
                best_lr=Lrs[j]
                best_p_count=int(PointCounts[k])

print(best_para,best_p_count,best_lr,min_rmse)


# 1.c
data_training_val_unproc = air_quality.iloc[index[0:N + Nval], :].copy()  # Select the validation data

data_training_val_unproc,avers,stds=DealtMissingValue(data_training_val_unproc)

colNames=list(data_training_val_unproc)
yTrain_val=  data_training_val_unproc[colNames[0]].values
yTrain_val=yTrain_val.reshape(len(yTrain_val),1)
XTrain_val = data_training_val_unproc.drop(columns=colNames[0]).values

# 1.d
data_test_unproc,avers_,stds_=DealtMissingValue(data_test_unproc)
colNames=list(data_test_unproc)
yTest=  data_test_unproc[colNames[0]].values
yTest=yTest.reshape(len(yTest),1)
XTest = data_test_unproc.drop(columns=colNames[0]).values

w=train_single_group(XTrain_val,yTrain_val,200,best_para,best_lr,best_p_count)

print(RMSE(add_1s_before_x(XTrain_val), yTrain_val, w))
print(R2(add_1s_before_x(XTrain_val),yTrain_val,w))

print(RMSE(add_1s_before_x(XTest), yTest, w))
print(R2(add_1s_before_x(XTest),yTest,w))
