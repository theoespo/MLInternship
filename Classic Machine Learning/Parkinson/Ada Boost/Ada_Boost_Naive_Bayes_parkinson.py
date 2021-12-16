from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
from math import inf
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

n_trees = [30, 40, 50, 60, 70, 80, 90, 100]

max_acc = 0
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

for i in range(len(n_trees)):

    print("* computing for n_trees =", n_trees[i], "*")
    a = time.process_time()
    model = AdaBoostClassifier(n_estimators=n_trees[i], base_estimator=GaussianNB())
    model.fit(X_train, y_train)
    predicted_target = model.predict(X_test)

    current_acc = accuracy_score(y_test, predicted_target)
    b = time.process_time()

    if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time): 
        max_acc = current_acc
        max_parameter = n_trees[i]
        max_time = b - a

print("\n===============================================================================================\n")
print("             Best model with Ada Boost based on Naive Bayes :\n\nNumber of estimators =", max_parameter, "\nBest accuracy :", max_acc, "\nTime to process :", max_time)