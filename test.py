

# import pandas as pd
# import numpy as np
# from pandas import DataFrame
# q=np.ones([1,3])
# print(type(q))
# print(q.shape)
# w=np.ones([3,1])
# print(type(w))
# print(w.shape)
# qw=np.dot(q,w)
# print(type(qw))
# print(qw.shape)
#
# print(w)
# w_qw=w-1
# print(type(w_qw))
# print(w_qw.shape)
# print(w_qw)

import numpy as np
from sklearn.model_selection import PredefinedSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,6]])
y = np.array([0, 0, 1, 1,1])

# test_fold=np.ones(5)
test_fold= np.zeros(3).tolist()

test_fold= np.hstack((np.ones(3)*-1,np.zeros(2)))
print(test_fold)

ps = PredefinedSplit(test_fold)
# print(ps.get_n_splits())

print(ps)
a=[ 0,  1, -1,  1]
PredefinedSplit(test_fold=a)
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
