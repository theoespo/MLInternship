from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import preprocessing, tree
import numpy as np
import time
from math import inf
from sklearn.ensemble import AdaBoostClassifier
import pickle

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

mean_len = 10
n_trees = [30, 40, 50, 60, 70, 80, 90, 100]
MSS = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
criterion_ = ["entropy", "gini"]

max_acc = 0
max_roc = 0
max_parameter = (0, 0, 0)
max_time = inf

best_train = ([], [])


#############################################################################################################################################
# AB with DT using all data


file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

data = np.array(num_list2(data[1:]))
target = np.array(num_list(target[1:]))


##############################################################################################################################################
# AB with DT using sliced data

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
# AB with DT using data and PCA

"""
file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:2] + l[4:5] + l[9:10] + l[15:16] + l[18:21])
    target.append(l[17])

data = PCA(n_components=6).fit_transform(num_list2(data[1:]))
"""

##############################################################################################################################################

##############################################################################################################################################
# single split of the data + KFold


data2, X_test, target2, y_test = train_test_split(data, target, test_size=0.20, random_state=33)

scaler = preprocessing.StandardScaler().fit(data2)
data2 = scaler.transform(data2)
X_test = scaler.transform(X_test)

kfold = KFold(n_splits=10)


##############################################################################################################################################


for k in range(len(n_trees)):
    print("* computing n_trees =", n_trees[k], "*")
    a2 = time.process_time()
    for i in range(len(MSS)):
        for j in range(len(criterion_)):

            a = time.process_time()
            train_acc = 0
            for train_index, test_index in kfold.split(data2):
                X_train, y_train = data2[train_index], target2[train_index]
                X_tt, y_tt = data[test_index], target[test_index]
                model = AdaBoostClassifier(n_estimators=n_trees[k], base_estimator=tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[j]))
                model.fit(X_train, y_train)
                predicted_target_train = model.predict(X_tt)
                current_train_acc = accuracy_score(y_tt, predicted_target_train)
                if train_acc < current_train_acc:
                    pickle.dump(model, open("temp_Ada_boost_train.sav", "wb"))

            model = pickle.load(open("temp_Ada_boost_train.sav", "rb"))
            predicted_target = model.predict(X_test)
            b = time.process_time()

            current_acc = accuracy_score(y_test, predicted_target)
            current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))

            if max_acc < current_acc or (max_acc == current_acc and b - a < max_time):
                max_acc = current_acc
                max_parameter = n_trees[k], MSS[i], criterion_[j]
                max_time = b - a
                pickle.dump(model, open("Ada_Boost_DT_best_model.sav", "wb"))


            if max_roc < current_roc:
                max_roc = current_roc


print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree : \n\nNumber of estimators =", max_parameter[0],"\nMinimum samples split =", max_parameter[1], ", criterion :", max_parameter[2],"\nAccuracy :", truncate(max_acc, 5), "\nTime to process :", max_time)
print("\nMaximum ROC AUC score :", max_roc)


