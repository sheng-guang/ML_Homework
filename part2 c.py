import sys

import matplotlib.pyplot as plt


def display(a):
    pass
import torch

print( torch.cuda.is_available())


# import packages
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
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



# get dataset info
data_flag = 'octmnist'
info = INFO[data_flag]

DataClass = getattr(medmnist, info['python_class'])
# preprocessing
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
# load data and transform
train_dataset = DataClass(split='train',transform=data_transform, download=True)
val_dataset = DataClass(split='val',transform=data_transform, download=True)
test_dataset = DataClass(split='test',transform=data_transform, download=True)

# encapsulate data into dataloader form
BATCH_SIZE = 128
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
val_loader=data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# def train score function
use_gpu = torch.cuda.is_available()
import time
def train_model(model,times=1):
    begin = time.perf_counter()
    # define loss function and optimizer
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
    for i in range(times):
        model.train()
        for inputs, targets in tqdm(train_loader):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            # sys.exit()
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            if use_gpu:
                loss = loss.cpu()
            loss.backward()
            optimizer.step()
    end = time.perf_counter()
    return end-begin

# evaluation
def score(model,split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    if use_gpu:
        y_true, y_score = y_true.cuda(), y_score.cuda()

    if split=='train':
        data_loader=train_loader_at_eval
    elif split=='val':
        data_loader=val_loader
    else:
        data_loader=test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
        if (use_gpu):
            y_true = y_true.cpu()
            y_score = y_score.cpu()
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(data_flag, split)

        metrics = evaluator.evaluate(y_score)
        print(split,',acc=',metrics.ACC,',auc=',metrics.AUC)
        return metrics.ACC,metrics.AUC


# def cnn models
class Net22(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net22, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(16, 64, 3,1,1),nn.BatchNorm2d(64),nn.MaxPool2d(2),nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(64 * 14 * 14, 128),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Net22_2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net22_2, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,16,(2,4),1,1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(16, 32, (2,4),1,1),nn.BatchNorm2d(32),nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(32 * 30 * 26, 16),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16, num_classes))

    def forward(self, x):
        # print(0,x.size())
        x = self.layer1(x)
        # print(1,x.size())
        x = self.layer2(x)
        # print(2,x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(11, x.size())
        x = self.fc2(x)
        # print(22,x.size())
        return x

class Net22_3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net22_3, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,16,2,1,0),nn.BatchNorm2d(16), nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(16, 16,2,1,1),nn.BatchNorm2d(16),nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(16 * 28 * 28, 16),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16, num_classes))

    def forward(self, x):
        # print(0,x.size())
        x = self.layer1(x)
        # print(1,x.size())
        x = self.layer2(x)
        # print(2,x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(11, x.size())
        x = self.fc2(x)
        # print(22,x.size())
        return x

# -------------------------------------------------
# train and choise the best model
n_channels = info['n_channels']
n_classes = len(info['label'])

model22_1 = Net22(in_channels=n_channels, num_classes=n_classes)
model22_2 = Net22_2(in_channels=n_channels,num_classes=n_classes)
model22_3 = Net22_3(in_channels=n_channels,num_classes=n_classes)
modesls=[model22_1,model22_2,model22_3]
best_m2=0

for model in modesls:
    print(type(model))
    time_cost=train_model(model)
    acc,auc = score(model, 'val')
    model.M2=acc
    model.M4=time_cost
    if acc>best_m2:
        best_model=model
        best_m2=acc

#         -------------------------------------------
# print best score
def print_model_score(best_model):
    M1,M1_ = score(best_model, 'train')
    best_model.M1=M1
    M3,m3_ = score(best_model, 'test')
    best_model.M3=M3
    print("model ",type(best_model))
    print("M1=",best_model.M1,'  m2=',best_model.M2,'  m3=',best_model.M3,'  m4=',best_model.M4)
print("mormal")
for model in modesls:
    if model==best_model:continue
    print_model_score(model)
print("")
print("best")
print_model_score(best_model)


# def 3 3Conv and 3FC cnn model
class Net33_1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net33_1, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,64,(3,3)),nn.BatchNorm2d(64),nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(64, 128, (3,3)),nn.BatchNorm2d(128),nn.ReLU())
        self.layer3=nn.Sequential(nn.Conv2d(128, 256, (3,3)),nn.BatchNorm2d(256),nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(256 * 22 * 22, 256),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 64),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        # print(1,x.size())
        x = self.layer2(x)
        # print(2,x.size())
        x = self.layer3(x)
        # print(3,x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(11, x.size())
        x = self.fc2(x)
        # print(22,x.size())
        x = self.fc3(x)
        return x

class Net33_2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net33_2, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,16,(3,3),1,1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(16, 32, (3,3)),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.layer3=nn.Sequential(nn.Conv2d(32, 64, (3,3),1,1),nn.BatchNorm2d(64),nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(64 * 13 * 13, 256),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 64),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        # print(1,x.size())
        x = self.layer2(x)
        # print(2,x.size())
        x = self.layer3(x)
        # print(3,x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(11, x.size())
        x = self.fc2(x)
        # print(22,x.size())
        x = self.fc3(x)
        return x

class Net33_3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net33_3, self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(in_channels,16,(3,3)),nn.BatchNorm2d(16), nn.ReLU())
        self.layer2=nn.Sequential(nn.Conv2d(16, 64,(3,3)),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.layer3=nn.Sequential(nn.Conv2d(64, 128,(3,3),1,1),nn.BatchNorm2d(128),nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(128 * 12 * 12, 256),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 64),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        # print(1,x.size())
        x = self.layer2(x)
        # print(2,x.size())
        x = self.layer3(x)
        # print(3,x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(11, x.size())
        x = self.fc2(x)
        # print(22,x.size())
        x = self.fc3(x)
        return x

# train and choise the best model
n_channels = info['n_channels']
n_classes = len(info['label'])
#
model33_1 = Net33_1(in_channels=n_channels, num_classes=n_classes)
model33_2 = Net33_2(in_channels=n_channels,num_classes=n_classes)
model33_3 = Net33_3(in_channels=n_channels,num_classes=n_classes)
modesls=[model33_1,model33_2,model33_3]
best_m2=0

for model in modesls:
    print(type(model))
    time_cost=train_model(model)
    acc,auc = score(model, 'val')
    model.M2=acc
    model.M4=time_cost
    if acc>best_m2:
        best_33model=model
        best_m2=acc


# show report
def print_model_score(m):
    M1,M1_ = score(m, 'train')
    m.M1=M1
    M3,m3_ = score(m, 'test')
    m.M3=M3
    print("model ",type(m))
    print("M1=",m.M1,'  m2=',m.M2,'  m3=',m.M3,'  m4=',m.M4)
print("normal")
for model in modesls:
    if model==best_33model:continue
    print_model_score(model)
print("")
print("best")
print_model_score(best_33model)

import matplotlib.pyplot as plt
plt.Figure()
plt.scatter([1,2,3,4],[0,best_model.M1,best_model.M2,best_model.M3,best_model.M4],c="red",label="22 model")
plt.scatter([1.1,2.1,3.1,4.1],[0,best_33model.M1,best_33model.M2,best_33model.M3,best_33model.M4],c="red",label="33 model")
plt.legend()
plt.title("show M1 to M4")
plt.show()










# class Net(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Net, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU())
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(16, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes))
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#
# model = Net(in_channels=n_channels, num_classes=n_classes)
#
# train_model(model)
# time_cost=train_model(model)
# acc,auc = score(model, 'val')
# model.M2=acc
# model.M4=time_cost
#
# print_model_score(model)



