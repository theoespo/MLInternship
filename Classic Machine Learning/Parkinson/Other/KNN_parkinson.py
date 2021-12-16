from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing
import time
from math import inf
from sklearn.neighbors import KNeighborsClassifier

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

neighbors_ = [3, 5, 8, 10, 15, 20, 25, 30]
weights_ = ["uniform", "distance"]
algorithm_ = ["ball_tree", "kd_tree", "brute"]
leaf_size_ = [10, 20, 30, 40, 50, 60, 70]

max_acc = 0
max_roc = 0
max_parameter_acc = (-1, -1, -1, -1)
max_parameter_roc = (-1, -1, -1, -1)
max_time_acc = inf
max_time_roc = inf

#########################################################################################################################################
# KNN using all the data

"""
file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])
"""

##############################################################################################################################################
# KNN using sliced data

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
# KNN using 3 features


file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:2] + l[18:21])
    target.append(l[17])


##############################################################################################################################################

X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i in range(len(neighbors_)):

    for j in range(len(weights_)):

        for k in range(len(algorithm_)):

            for l in range(len(leaf_size_)):

                a3 = time.process_time()
                model = KNeighborsClassifier(neighbors_[i], weights=weights_[j], algorithm=algorithm_[k])
                model.fit(X_train, y_train)
                predicted_target = model.predict(X_test)

                current_acc = accuracy_score(num_list(y_test), num_list(predicted_target))
                current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))
                b3 = time.process_time()

                if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time_acc): 
                    max_acc = current_acc
                    max_parameter_acc = (neighbors_[i], weights_[j], algorithm_[k], leaf_size_[l])
                    max_time_acc = b3 - a3
                
                if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
                    max_roc = current_roc
                    max_parameter_roc = (neighbors_[i], weights_[j], algorithm_[k], leaf_size_[l])
                    max_time_roc = b3 - a3

print("\n===============================================================================================\n")
print("             Best model with KNN (accuracy) :\nNumber of Neighbors =", max_parameter_acc[0], ", Weights =", max_parameter_acc[1], ", algorithm :", max_parameter_acc[2], ", leaf size :", max_parameter_acc[3], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc)
print("\n===============================================================================================\n")
print("             Best model with KNN (ROC) :\nNumber of Neighbors =", max_parameter_roc[0], ", Weights =", max_parameter_roc[1], ", algorithm :", max_parameter_roc[2], ", leaf size :", max_parameter_acc[3], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

