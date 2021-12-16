from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
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

C_ = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]
gamma_ = [0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000]
kernel_ = ['linear', 'rbf', 'poly', 'sigmoid']

max_acc = 0
max_roc = 0
max_parameter = (-1, -1, -1)
max_C = inf
max_gamma = inf
max_time = inf

file = open("parkinsons.data")
data = []
target = []
for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


for i in range(len(C_)):
    a = time.process_time()
    for j in range(len(gamma_)):
        a2 = time.process_time()
        for k in range(len(kernel_)):
            
            a3 = time.process_time()
            estimator = SVC(kernel='linear', gamma=gamma_[j],C=C_[i])
            estimator.fit(X_train, y_train)
            model = RFE(estimator)
            model.fit(X_train, y_train)
            predicted_target = model.predict(X_test)

            current_acc = accuracy_score(y_test, predicted_target)
            current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))
            b3 = time.process_time()

            if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time): 
                max_acc = current_acc
                max_parameter = (C_[i], gamma_[j], kernel_[k])
                max_time = b3 - a3
            
            if max_roc < current_roc:
                max_roc = current_roc

        b2 = time.process_time()
        if C_[i] == 1000:
            print("time for gamma =", gamma_[j], ":", b2 - a2)
    b = time.process_time()
    print("time with C =", C_[i], ":", b - a)

print("\n===============================================================================================\n")
print("             Best model with SVM :\nC =", max_parameter[0], ", gamma =", max_parameter[1], ", kernel :", max_parameter[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time)
print("Best ROC AUC score :", max_roc)