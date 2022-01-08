from numpy.core.fromnumeric import mean
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score
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

n_trees = [30, 40, 50, 60, 70, 80, 90, 100]
MSS = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
criterion_ = ["entropy", "gini"]

kfold_param = 5
filename = os.path.basename(__file__)[:-18]
path_model = Path("model")
path_perf = Path("perf")

f = open(path_perf / ("perf_" + filename + str(kfold_param) + "-Fold_best_model.txt"), "a")
f.close()
f = open(path_perf / ("perf_" + filename + str(kfold_param) + "-Fold_best_model.txt"))
    
param = []
for line in f:
    l = line.split(",")
    for i in range(len(l)):
        param += [float(l[i])]
f.close()

# if len(param) != 0:
#     max_acc = param[0]
#     max_time = param[1]
# else:
max_acc = 0
max_time = inf
max_parameter = (0, 0, 0)

max_roc = 0
max_parameter_roc = (0, 0, 0)
max_time_roc = inf


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


kfold = KFold(n_splits=kfold_param, shuffle=True, random_state=33)
mean_split = 0


##############################################################################################################################################


for k in range(len(n_trees)):

    print("* computing for n_trees =", n_trees[k], "*")
    for i in range(len(MSS)):

        for j in range(len(criterion_)):

            max_param_prev = max_parameter
            temp_mean_split = 0
            max_param_prev_roc = max_parameter_roc
            temp_mean_split_roc = 0
            all_acc = np.zeros(shape=(5,))
            all_roc = np.zeros(shape=(5,))
            c = 0
            
            for train_index, test_index in kfold.split(data):

                X_train, y_train = data[train_index], target[train_index]
                X_test, y_test = data[test_index], target[test_index]

                a = time.process_time()
                model = AdaBoostClassifier(n_estimators=n_trees[k], base_estimator=tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[j]))
                model.fit(X_train, y_train)
                predicted_target = model.predict(X_test)

                current_acc = accuracy_score(y_test, predicted_target)
                current_roc = roc_auc_score(y_test, predicted_target)
                temp_mean_split_roc += current_roc
                temp_mean_split += current_acc
                all_acc[c] = current_acc
                all_roc[c] = current_roc
                c += 1

                b = time.process_time()

                if max_acc < current_acc or (max_acc == current_acc and b - a < max_time):
                    max_acc = current_acc
                    max_parameter = n_trees[k], MSS[i], criterion_[j]
                    max_time = b - a
                    pickle.dump(model, open(path_model / (filename  + str(kfold_param) +"-Fold_Best_Model.sav"), "wb"))
                    f = open(path_perf / ("perf_" + filename + str(kfold_param) + "-Fold_best_model.txt"), "w")
                    f.write(str(max_acc) + ", " + str(max_time))
                    f.close()
                
                if max_roc < current_roc or (max_roc == current_roc and max_time_roc > b - a):
                    max_roc = current_roc
                    max_parameter_roc = n_trees[k], MSS[i], criterion_[j]
                    max_time_roc = b - a
            
            temp_mean_split = temp_mean_split/kfold_param
            if max_param_prev != max_parameter:
                mean_split = temp_mean_split
                std_acc = all_acc.std()
                print(all_acc)

            temp_mean_split_roc = temp_mean_split_roc/kfold_param
            if max_param_prev_roc != max_parameter_roc:
                mean_split_roc = temp_mean_split_roc
                std_roc = all_roc.std()




print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree : \n\nNumber of estimators =", max_parameter[0],"\nMinimum samples split =", max_parameter[1], ", criterion :", max_parameter[2],"\nAccuracy :", truncate(max_acc, 5), "\nTime to process :", max_time)
print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree : \n\nNumber of estimators =", max_parameter_roc[0],"\nMinimum samples split =", max_parameter_roc[1], ", criterion :", max_parameter_roc[2],"\nAccuracy :", max_roc, "\nTime to process :", max_time_roc)
print("\nAverage on all splits of the best model (acc):", mean_split, "+/-", std_acc)
print("\nAverage on all splits of the best model (roc):", mean_split_roc, "+/-", std_roc)

