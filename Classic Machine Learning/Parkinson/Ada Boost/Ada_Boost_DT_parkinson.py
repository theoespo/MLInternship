from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn import preprocessing, tree
import numpy as np
import time
from math import inf
from sklearn.ensemble import AdaBoostClassifier

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

max_acc = [0, 0]
max_roc = 0
max_parameter_acc = [(0, 0, 0), (0, 0, 0)]
max_time_acc = [inf, inf]
max_parameter_roc = (0, 0, 0)
max_time_roc = inf

mean_max = 0
all_max_parameter = []
weight_max_parameter = []
average_max_accuracy = []

#############################################################################################################################################
# AB with DT using all data


file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])


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
# single split of the data
X_train, X_test, y_train, y_test = train_test_split(data, target[1:], test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


for l in range(mean_len):

    print("* computing", str(l+1) + "/" + str(mean_len), "*")
    max_acc[0] = 0
    for k in range(len(n_trees)):

        for i in range(len(MSS)):

            for j in range(len(criterion_)):

                a = time.process_time()
                model = AdaBoostClassifier(n_estimators=n_trees[k], base_estimator=tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[j]))
                model.fit(X_train, y_train)
                predicted_target = model.predict(X_test)
                b = time.process_time()

                current_acc = accuracy_score(y_test, predicted_target)
                current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))

                if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                    max_acc[0] = current_acc
                    max_parameter_acc[0] = n_trees[k], MSS[i], criterion_[j]
                    max_time_acc[0] = b - a

                if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                    max_acc[1] = current_acc
                    max_parameter_acc[1] = n_trees[k], MSS[i], criterion_[j]
                    max_time_acc[1] = b - a

                if max_roc < current_roc:
                    max_roc = current_roc
                    max_parameter_roc = n_trees[k], MSS[i], criterion_[j]
                    max_time_roc = b - a

    
    mean_max += max_acc[0]
    if not(max_parameter_acc[0] in all_max_parameter):
        all_max_parameter.append(max_parameter_acc[0])
        weight_max_parameter.append(1)
        average_max_accuracy.append(max_acc[0])
    else:
        weight_max_parameter[all_max_parameter.index(max_parameter_acc[0])] += 1
        average_max_accuracy[all_max_parameter.index(max_parameter_acc[0])] += max_acc[0]

mean_max = mean_max/mean_len
for i in range(len(average_max_accuracy)):
    average_max_accuracy[i] = average_max_accuracy[i]/weight_max_parameter[i]
for i in range(len(weight_max_parameter)):
    weight_max_parameter[i] = weight_max_parameter[i]/(mean_len/100)

print("\n* Average of maximum accuracy :", truncate(mean_max, 5), "\n* Parameters giving a maximum of accuracy :", all_max_parameter, "\n--> with a percentage of", weight_max_parameter, "\n--> with an average accuracy of :", average_max_accuracy)
print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree (accuracy) : \n\nNumber of estimators =", max_parameter_acc[1][0],"\nMinimum samples split =", max_parameter_acc[1][1], ", criterion :", max_parameter_acc[1][2],"\nAccuracy :", truncate(max_acc[1], 5), "\nTime to process :", max_time_acc[1])
print("\n===================================================================================================================================\n")
print("          Best model with Ada Boost based on Decision Tree (ROC AUC) : \n\nNumber of estimators =", max_parameter_roc[0],"\nMinimum samples split =", max_parameter_roc[1], ", criterion :", max_parameter_roc[2],"\nAccuracy :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)


