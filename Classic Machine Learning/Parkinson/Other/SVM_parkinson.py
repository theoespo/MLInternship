from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing, svm
import time
from math import inf

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(float(data[i]))
    return data_num

C_ = [0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]
gamma_ = [0.0001, 0.001, 0.01, 1, 10, 100, 1000]
kernel_ = ['linear', 'rbf', 'poly', 'sigmoid']

max_acc = 0
max_roc = 0
max_parameter_acc = (-1, -1, -1)
max_parameter_roc = (-1, -1, -1)
max_time_acc = inf
max_time_roc = inf

#########################################################################################################################################
# SVM using all the data


file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])


##############################################################################################################################################
# SVM using sliced data

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
# SVM using 3 features

"""
file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:2] + l[18:20])
    target.append(l[17])
"""

##############################################################################################################################################

##############################################################################################################################################
# single split of the data
X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##############################################################################################################################################

for i in range(len(C_)):
    a = time.process_time()
    for j in range(len(gamma_)):
        a2 = time.process_time()
        for k in range(len(kernel_)):
            
            a3 = time.process_time()
            model = svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j])
            model.fit(X_train, y_train)
            predicted_target = model.predict(X_test)

            current_acc = accuracy_score(y_test, predicted_target)
            current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))
            b3 = time.process_time()

            if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time_acc): 
                max_acc = current_acc
                max_parameter_acc = (C_[i], gamma_[j], kernel_[k])
                max_time_acc = b3 - a3
            
            if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
                max_roc = current_roc
                max_parameter_roc = (C_[i], gamma_[j], kernel_[k])
                max_time_roc = b3 - a3

        b2 = time.process_time()
        if C_[i] == 1000:
            print("time for gamma =", gamma_[j], ":", b2 - a2)
    b = time.process_time()
    print("time with C =", C_[i], ":", b - a)

print("\n===============================================================================================\n")
print("             Best model with SVM (accuracy) :\nC =", max_parameter_acc[0], ", gamma =", max_parameter_acc[1], ", kernel :", max_parameter_acc[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc)
print("\n===============================================================================================\n")
print("             Best model with SVM (ROC) :\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

