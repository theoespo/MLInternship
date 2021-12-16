from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, svm
import time
from math import inf
from sklearn.ensemble import AdaBoostClassifier

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

n_trees = [30, 40, 50, 60, 70, 80, 90, 100]
C_ = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100]
gamma_ = [0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000]
kernel_ = ['linear', 'rbf', 'poly', 'sigmoid']

max_acc = 0
max_parameter = (-1, -1, -1, -1)
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

for n in range(len(n_trees)):
    print("* computing for n_trees =", n_trees[n], "*")
    for i in range(len(C_)):
        # a = time.process_time()
        for j in range(len(gamma_)):
            # a2 = time.process_time()
            for k in range(len(kernel_)):
                
                a3 = time.process_time()
                model = AdaBoostClassifier(n_estimators=n_trees[n], base_estimator=svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j]), algorithm="SAMME")
                model.fit(X_train, y_train)
                predicted_target = model.predict(X_test)

                current_acc = truncate(accuracy_score(y_test, predicted_target), 5)
                b3 = time.process_time()

                if (max_acc < current_acc) or (max_acc == current_acc and b3 - a3 < max_time): 
                    max_acc = current_acc
                    max_parameter = (C_[i], gamma_[j], kernel_[k], n_trees[n])
                    max_time = b3 - a3

        #     b2 = time.process_time()
        #     if C_[i] == 1000:
        #         print("time for gamma =", gamma_[j], ":", b2 - a2)
        # b = time.process_time()
        # print("time with C =", C_[i], ":", b - a)

print("\n===============================================================================================\n")
print("             Best model with Ada Boost based on SVM :\n\nNumber of estimators =", max_parameter[3], "\nC =", max_parameter[0], ", gamma =", max_parameter[1], ", kernel :", max_parameter[2], "\nBest accuracy :", max_acc, "\nTime to process :", max_time)