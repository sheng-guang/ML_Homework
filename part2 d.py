import sys

from torch import multiprocessing


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms

if __name__ == '__main__':
    batchSize=64
    classes = datasets.FashionMNIST.classes
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    #Load the training data
    trainset =datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform)
    testset = datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)
    #Load the test data
    def get_indices(dataset, class_name,class2):
        indices = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_name or dataset.targets[i]==class2:
                indices.append(i)
        return indices
    idx_train_0=get_indices(trainset,0,8)
    # print(len(idx_train_0))
    # print(idx_train_0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, num_workers=2,sampler=SubsetRandomSampler(idx_train_0))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=2)
    print('Training set size:', len(trainset))
    print('Test set size:',len(testset))




    first=True
    np_train=[]
    np_train_y=[]
    for i in idx_train_0:
        to= trainset.data[i].numpy().reshape([1,28*28])

        if first:
            np_train=to
            np_train_y=trainset.targets[i].numpy()
            first=False
        else:
            np_train=np.vstack((np_train,to))
            np_train_y = np.hstack((np_train_y, trainset.targets[i]))
    np_train=np_train/256
    print(np_train.shape)
    print(np_train_y.shape)


    from sklearn.decomposition import PCA
    pca = PCA()
    np_train_pca= pca.fit_transform(np_train)
    # print(pca.components_)

    # # show vector
    # ratio_top5=pca.explained_variance_ratio_[:5]
    # plt.scatter(range(5), ratio_top5)
    # plt.plot(range(5), ratio_top5)
    #
    # plt.title("ratio_top5")
    # plt.ylabel("ratio")
    #
    # plt.grid()
    # plt.show()


    def show_two_pic(old,new):
        old=old.reshape(28, 28)
        new=new.reshape(28, 28)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(old,cmap=plt.cm.gray, interpolation='nearest',clim=(0, 255));
        plt.title('image', fontsize=24)

        plt.subplot(1, 2, 2)
        plt.imshow(new,cmap=plt.cm.gray, interpolation='nearest',clim=(0, 255))
        plt.title('pca image', fontsize=24)
        plt.show()
    # for i in idx_train_0:
    #     count=0
    #     if trainset.targets[i]==0:
    #         count+=1
    #         show_two_pic(np_train[i], pca.inverse_transform(np_train_pca[i]))
    #     if count==5:
    #         break

    np_train_pca_01= np_train_pca[:,0:2]



    plt.scatter(np_train_pca_01[np_train_y==0,0],np_train_pca_01[np_train_y==0,1],c="blue",label='class 0')
    plt.scatter(np_train_pca_01[np_train_y==8,0],np_train_pca_01[np_train_y==8,1],c="orange",label='class 0')
    plt.legend()
    plt.title("show after pca")
    plt.show()



    print("start SC")

    np_train_pca_01= np_train_pca[:,0:2]

    from sklearn.preprocessing import StandardScaler
    np_train_pca_01=StandardScaler().fit_transform(np_train_pca_01)


    from sklearn.cluster import SpectralClustering

    print(np_train_pca_01.shape)
    pred_y = SpectralClustering(n_clusters=2).fit_predict(np_train_pca_01,y=None)
    print(pred_y)
    # plt.scatter(np_train_pca_01[pred_y==0,0],np_train_pca[np_train_y==0,1],c="blue",label='class 0')
    # plt.scatter(np_train_pca[np_train_y==8,0],np_train_pca[np_train_y==8,1],c="orange",label='class 0')
    # plt.legend()
    # plt.title("show after pca")
    # plt.show()