train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader=data.DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


def split_x_y(loader):
    # len
    length=len(loader.dataset)
    # x wide
    shape=np.array( loader.dataset[0][0][0]).shape
    imglen=shape[0]*shape[1]

    ylist=np.array(range(length))
    for i in range(length):
        datas=loader.dataset[i]
        x = np.array(datas[0][0]).reshape([1, imglen])
        y = datas[1][0]
        if i==0:
            xlist=x
        else:
            xlist=np.vstack((xlist,x))
        ylist[i]=y
    print(xlist.shape)
    print(ylist.shape)
    return xlist,ylist

x_train,y_train=split_x_y(train_loader)
x_val,y_val=split_x_y(val_loader)
x_test,y_test=split_x_y(test_loader)