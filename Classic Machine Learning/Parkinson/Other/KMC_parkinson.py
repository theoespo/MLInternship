from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing
import time
from math import inf
from sklearn.cluster import KMeans

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

init_ = ['k-means++', 'random']
algorithm_ = ["full", "elkan",]

max_acc = 0
max_roc = 0
max_parameter_acc = (-1, -1)
max_parameter_roc = (-1, -1)
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
# KNN using 4 features


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

for i in range(len(init_)):

        for j in range(len(algorithm_)):

                a3 = time.process_time()
                model = KMeans(2, init=init_[i], n_init=500, algorithm=algorithm_[j])
                model.fit(X_train)
                predicted_target = model.predict(X_test)

                current_acc = accuracy_score(num_list(y_test), num_list(predicted_target))
                current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))
                b3 = time.process_time()

                if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time_acc): 
                    max_acc = current_acc
                    max_parameter_acc = (init_[i], algorithm_[j])
                    max_time_acc = b3 - a3
                
                if max_roc < current_roc or (max_roc == current_roc and b3 - a3 < max_time_roc):
                    max_roc = current_roc
                    max_parameter_roc = (init_[i], algorithm_[j])
                    max_time_roc = b3 - a3

print("\n===============================================================================================\n")
print("             Best model with KMC (accuracy) :\nInit method =", max_parameter_acc[0], ", algorithm :", max_parameter_acc[1], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc)
print("\n===============================================================================================\n")
print("             Best model with KMC (ROC) :\nInit method =", max_parameter_roc[0], ", algorithm :", max_parameter_roc[1], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

