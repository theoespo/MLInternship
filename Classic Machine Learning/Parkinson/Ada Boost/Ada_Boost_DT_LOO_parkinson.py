from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import tree
import time, pickle, os
import numpy as np
from math import inf
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

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

filename = os.path.basename(__file__)[:-16]
path_model = Path("model")
path_perf = Path("perf")

f = open(path_perf / ("perf_" + filename + "LOO_best_model.txt"), "a")
f.close()
f = open(path_perf / ("perf_" + filename + "LOO_best_model.txt"))
    
param = []
for line in f:
    l = line.split(",")
    for i in range(len(l)):
        param += [float(l[i])]
f.close()

if len(param) != 0:
    max_acc = param[0]
    max_time = param[1]
else:
    max_acc = 0
    max_time = inf


#############################################################################################################################################
# AB with DT using all data


file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])
file.close()

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
# splitting the data using KFold


loo = LeaveOneOut()
mean_split = 0


##############################################################################################################################################

count = 0
for train_index, test_index in loo.split(data):

    count += 1
    print("* computing", count, "/ 195 *")
    X_train, y_train = data[train_index], target[train_index]
    X_test, y_test = data[test_index], target[test_index]

    a = time.process_time()
    model = AdaBoostClassifier(n_estimators=40 ,base_estimator=tree.DecisionTreeClassifier(min_samples_split=40, criterion="entropy"))
    model.fit(X_train, y_train)
    predicted_target = model.predict(X_test)

    current_acc = accuracy_score(y_test, predicted_target)
    mean_split += current_acc
    b = time.process_time()

    if max_acc < current_acc or (max_acc == current_acc and b - a < max_time):
        max_acc = current_acc
        max_time = b - a
        pickle.dump(model, open(path_model / (filename  + "LOO_Best_Model.sav"), "wb"))
        f = open(path_perf / ("perf_" + filename + "LOO_best_model.txt"), "w")
        f.write(str(max_acc) + ", " + str(max_time))
        f.close()
            
mean_split = mean_split/195



print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree :\nAccuracy :", truncate(max_acc, 5), "\nTime to process :", max_time)
print("\nAverage on all splits of the best model :", mean_split)

