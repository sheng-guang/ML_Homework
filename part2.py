import sys

def display(a):
    pass


# import packages
import numpy as np
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import torch
# A
# set random seed
np.random.seed(1771480)#001771480
torch.manual_seed(1771480)


# B1
# show pictures for each class

for key in INFO.keys():
    break

    print(key)
    if key.endswith('3d'): continue
    info = INFO[key]
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', download=True)
    val_dataset = DataClass(split='val', download=True)
    test_dataset = DataClass(split='test', download=True)
    print("train:")
    display(train_dataset.montage(length=5))
    print("val:")
    display(val_dataset.montage(length=4))
    print("test:")
    display(test_dataset.montage(length=4))
#

# sys.exit()



# b2
# get dataset info
info = INFO['breastmnist']
DataClass = getattr(medmnist, info['python_class'])
# preprocessing
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
# load data and transform
train_dataset = DataClass(split='train',transform=data_transform, download=True)
val_dataset = DataClass(split='val',transform=data_transform, download=True)
test_dataset = DataClass(split='test',transform=data_transform, download=True)

def split_x_y(dataset):
    # len
    length=len(dataset)
    # x wide
    shape=np.array(dataset[0][0][0]).shape
    imglen=shape[0]*shape[1]

    ylist=np.array(range(length))
    for i in range(length):
        datas=dataset[i]
        x = np.array(datas[0][0]).reshape([1, imglen])
        y = datas[1][0]
        if i==0:
            xlist=x
        else:
            xlist=np.vstack((xlist,x))
        ylist[i]=y
    return xlist,ylist

x_train,y_train=split_x_y(train_dataset)
x_val,y_val=split_x_y(val_dataset)
x_test,y_test=split_x_y(test_dataset)
x_train_val=np.vstack((x_train,x_val))
y_train_val=np.hstack((y_train,y_val))


# # train
# from sklearn.linear_model import LogisticRegression
# clist=np.linspace(0.001,0.3,5)
#
# max_M2=0
# best_c=0
# for c in clist:
#     logistic_regression = LogisticRegression(penalty="l2", max_iter=10000, C=c)
#     logistic_regression.fit(x_train,y_train)
#     M2=logistic_regression.score(x_val,y_val)
#     if M2>max_M2:
#         best_c=c
#         max_M2=M2
#
# print("best lr2 ",best_c)
#
# print("-------")
# logistic_regression = LogisticRegression(penalty="l2", max_iter=10000, C=best_c)
#
# logistic_regression.fit(x_train_val,y_train_val)
#
# M1 = logistic_regression.score(x_train, y_train)
#
# M3=logistic_regression.score(x_test,y_test)
# print("M1:",M1)
# print("M2:",max_M2)
# print("M3:",M3)



# ----------------------------------------------

def np2torch(x,y):
    x=torch.from_numpy(x.astype(np.float32))
    yline=torch.from_numpy(y.astype(np.float32).reshape(1,(len(y))))
    y=torch.from_numpy(y.astype(np.float32).reshape((len(y),1)))
    return x,y,yline

x_train_t,y_train_t,y_train_t_line=np2torch(x_train,y_train)
x_val_t,y_val_t,y_val_t_line=np2torch(x_val,y_val)
x_test_t,y_test_t,y_test_t_line=np2torch(x_test,y_test)
x_train_val_t,y_train_val_t,y_train_val_t_line=np2torch(x_train_val,y_train_val)


# ----------------------------------------------

def getscore(y_pre,y):
    correct_sum = y_pre.eq(y).sum()
    return correct_sum.item() / len(y_pre)
def model_getscore(model,x,y):
    y_pre = model(x).round().squeeze()
    return getscore(y_pre,y)

class LogisticRegression(torch.nn.Module):
    def __init__(self,input_features):
        super().__init__()
        self.linear=torch.nn.Linear(input_features,1)
    def forward(self,x):
        y=self.linear(x)
        return torch.sigmoid(y)

def train_new_model(l2,x,y,times=300):
    input_features = 28 * 28
    model = LogisticRegression(input_features=input_features)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=l2)
    for i in range(times):
        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

# ----------------------------------------------
l2s=np.linspace(0.0001,0.1,5)
max_M2_=0
best_l2=0

for l2 in l2s:
    model=train_new_model(l2,x_train_t,y_train_t)

    M2 = model_getscore(model,x_val_t,y_val_t_line)

    if M2>max_M2_:
        best_l2=l2
        max_M2_=M2
print("best L2:",best_l2)
print("-------")
model=train_new_model(best_l2,x_train_val_t,y_train_val_t)
M1_=model_getscore(model,x_train_t,y_train_t_line)

M3_=model_getscore(model,x_test_t,y_test_t_line)
print("M1:",M1_)
print("M2:",max_M2_)
print("M3:",M3_)

import matplotlib.pyplot as plt

plt.Figure()
plt.scatter([1,2,3],[M1,max_M2,M3],c="blue",label="model 1")
plt.scatter([1.1,2.1,3.1],[M1_,max_M2_,M3_],c="orange",label="model 2")

plt.legend()
plt.xlabel("m1                                   m2                                 m3")
plt.title("show M1 to M3")
plt.show()















