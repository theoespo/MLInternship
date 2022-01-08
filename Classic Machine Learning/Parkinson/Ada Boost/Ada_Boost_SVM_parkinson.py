from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing, svm
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

n_trees = [30, 40, 50, 60, 70]
C_ = [0.01, 1, 10, 100]
gamma_ = [0.0001, 0.001, 0.01, 1, 10, 100,]
kernel_ = ['linear', 'rbf', 'poly']

max_acc = 0
max_parameter = (-1, -1, -1, -1)
max_C = inf
max_gamma = inf
max_time = inf

max_roc = 0
max_parameter_roc = (0, 0, 0)
max_time_roc = inf

##############################################################################################################################################

# data_type = 0
# file = open("parkinsons.data")
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     data.append(l[1:17] + l[18:])
#     target.append(l[17])

# data = num_list2(data[1:])
# target = num_list(target[1:])

# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

##############################################################################################################################################

data_type = 2
file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

data = num_list2(data[1:])
target = num_list(target[1:])

X_train = []
X_test = []
y_train = []
y_test_oh = []
y_test = []
for i in range(0, 25):

    X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

    scaler = preprocessing.StandardScaler().fit(X_train_small)
    X_train_small = scaler.transform(X_train_small)
    X_test_small = scaler.transform(X_test_small)
    # y_test_oh_small = one_hot_encode(y_test_small, 3)

    X_train.append(X_train_small)
    X_test.append(X_test_small)
    y_train.append(y_train_small)
    # y_test_oh.append(y_test_oh_small)
    y_test.append(y_test_small)

##############################################################################################################################################

##############################################################################################################################################

if data_type == 0:
    for n in range(len(n_trees)):
        print("* computing for n_trees =", n_trees[n], "*")
        for i in range(len(C_)):

            for j in range(len(gamma_)):

                for k in range(len(kernel_)):
                    
                    a3 = time.process_time()
                    model = AdaBoostClassifier(n_estimators=n_trees[n], base_estimator=svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j]), algorithm="SAMME")
                    model.fit(X_train, y_train)
                    predicted_target = model.predict(X_test)

                    current_acc = truncate(accuracy_score(y_test, predicted_target), 5)
                    current_roc = roc_auc_score(y_test, predicted_target)
                    b3 = time.process_time()

                    if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time): 
                        max_acc = current_acc
                        max_parameter = (C_[i], gamma_[j], kernel_[k], n_trees[n])
                        max_time = b3 - a3
                        
                    if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
                        max_roc = current_roc
                        max_parameter_roc = (C_[i], gamma_[j], kernel_[k], n_trees[n])
                        max_time_roc = b3 - a3



    print("\n===============================================================================================\n")
    print("             Best model with Ada Boost based on SVM :\n\nNumber of estimators =", max_parameter[3], "\nC =", max_parameter[0], ", gamma =", max_parameter[1], ", kernel :", max_parameter[2], "\nBest accuracy :", max_acc, "\nTime to process :", max_time)
    print("\n===============================================================================================\n")
    print("             Best model with Ada Boost based on SVM :\n\nNumber of estimators =", max_parameter_roc[3], "\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest accuracy :", max_roc, "\nTime to process :", max_time_roc)

if data_type == 2:
    for l in range(len(X_test)):

        print("* computing", str(l+1) + "/" + str(len(X_test)), "*")
        for n in range(len(n_trees)):

            print("* computing for n_trees =", n_trees[n], "*")
            for i in range(len(C_)):

                for j in range(len(gamma_)):

                    for k in range(len(kernel_)):
                        
                        a3 = time.process_time()
                        model = AdaBoostClassifier(n_estimators=n_trees[n], base_estimator=svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j]), algorithm="SAMME")
                        model.fit(X_train[l], y_train[l])
                        predicted_target = model.predict(X_test[l])

                        current_acc = truncate(accuracy_score(y_test[l], predicted_target), 5)
                        current_roc = roc_auc_score(y_test[l], predicted_target)
                        b3 = time.process_time()

                        if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time): 
                            max_acc = current_acc
                            max_parameter = (C_[i], gamma_[j], kernel_[k], n_trees[n])
                            max_time = b3 - a3
                            max_seed_acc = l
                            
                        if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
                            max_roc = current_roc
                            max_parameter_roc = (C_[i], gamma_[j], kernel_[k], n_trees[n])
                            max_time_roc = b3 - a3
                            max_seed_roc = l



    print("\n===============================================================================================\n")
    print("             Best model with Ada Boost based on SVM :\n\nNumber of estimators =", max_parameter[3], "\nC =", max_parameter[0], ", gamma =", max_parameter[1], ", kernel :", max_parameter[2], "\nBest accuracy :", max_acc, "\nTime to process :", max_time, "seed :", max_seed_acc)
    print("\n===============================================================================================\n")
    print("             Best model with Ada Boost based on SVM :\n\nNumber of estimators =", max_parameter_roc[3], "\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest accuracy :", max_roc, "\nTime to process :", max_time_roc, "seed :", max_seed_roc)