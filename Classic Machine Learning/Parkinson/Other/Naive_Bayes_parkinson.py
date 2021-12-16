from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
from sklearn.naive_bayes import GaussianNB

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

a = time.process_time()
model = GaussianNB()
model.fit(X_train, y_train)
predicted_target = model.predict(X_test)

current_acc = accuracy_score(y_test, predicted_target)
b = time.process_time()

print("Accuracy Gaussian Naive Bayes method :", current_acc, "\nTime to process :", b - a)

