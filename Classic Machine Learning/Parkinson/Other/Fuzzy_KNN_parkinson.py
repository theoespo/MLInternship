from numpy.core.numeric import cross
from sklearn import neighbors
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
from math import inf
import numpy as np
from fknn import FuzzyKNN

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(float(data[i]))
    return data_num

def num_list2(data):
    data_num = [[] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_num[i].append(float(data[i][j]))
    return data_num

neighbors_ = [3, 5, 9, 11, 15, 21]

max_acc = 0
max_roc = 0
max_parameter_acc = -1
max_parameter_roc = -1
max_time_acc = inf
max_time_roc = inf

#########################################################################################################################################
# FuzzyKNN using all the data + PCA

"""
file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:-1] + [l[-1][:-1]])
    target.append(l[17])

data = PCA().fit_transform(data[1:])
# data = np.array(data[1:])
target = np.array(target[1:])
"""

##############################################################################################################################################
# FuzzyKNN using sliced data

"""
file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:2] + l[4:5] + l[9:10] + l[15:16] + l[18:21])
    target.append(l[17])
"""

##############################################################################################################################################
# FuzzyKNN using 3 features


file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:2] + l[18:21])
    target.append(l[17])

data = np.array(num_list2(data[1:]))
target = np.array(num_list(target[1:]))


##############################################################################################################################################

##############################################################################################################################################
# single split of the data
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

##############################################################################################################################################
# splitting the data using KFold
kfold = KFold(n_splits=10)


count = 1
scores = []
for train_index, test_index in kfold.split(data):
    print("* computing", count, "/ 10 *")
    X_train, y_train = data[train_index], target[train_index]
    X_test, y_test = data[test_index], target[test_index]
    count += 1

    for i in range(len(neighbors_)):

        a3 = time.process_time()
        model = FuzzyKNN(neighbors_[i])
        model.fit(X_train, y_train)
        predicted_target = model.predict(X_test)
        for j in range(len(predicted_target)):
            predicted_target[j] = predicted_target[j][0]

        current_acc = accuracy_score(y_test, predicted_target)
        # scores.append(cross_val_score(model, data, target, cv=10))
        # current_roc = roc_auc_score(num_list(target[test_index]), num_list(predicted_target))
        b3 = time.process_time()

        if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time_acc): 
            max_acc = current_acc
            max_parameter_acc = neighbors_[i]
            max_time_acc = b3 - a3
        
        # if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
        #     max_roc = current_roc
        #     max_parameter_roc = neighbors_[i]
        #     max_time_roc = b3 - a3



print("\n===============================================================================================\n")
print("             Best model with KNN (accuracy) :\nNumber of Neighbors =", max_parameter_acc, "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc)
# print(scores)
# print("\n===============================================================================================\n")
# print("             Best model with KNN (ROC) :\nNumber of Neighbors =", max_parameter_roc, "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

