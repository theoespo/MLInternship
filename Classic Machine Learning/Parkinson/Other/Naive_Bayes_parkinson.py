from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing
import time
from sklearn.naive_bayes import GaussianNB
from math import inf

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(float(data[i]))
    return data_num


def num_list2(data):
    data_num = [[] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] != '':
                data_num[i].append(float(data[i][j]))
            else:
                data_num[i].append(-1000)
    return data_num

max_acc = 0
max_roc = 0
max_time_acc = inf
max_time_roc = inf

file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

###################################################################################################################

# data_type = 0
# X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

###################################################################################################################

data_type = 2
data = num_list2(data[1:])
target = num_list(target[1:])

X_train = []
X_test = []
y_train = []
y_test_oh = []
y_test = []
for i in range(0, 200):

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

###################################################################################################################

if data_type == 0:
    a = time.process_time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    predicted_target = model.predict(X_test)

    current_acc = accuracy_score(y_test, predicted_target)
    current_roc = roc_auc_score(y_test, predicted_target)
    b = time.process_time()

    print("Accuracy Gaussian Naive Bayes method :", current_acc, "\nTime to process :", b - a)
    print("ROC Gaussian Naive Bayes method :", current_roc)

if data_type == 2:

    for i in range(len(X_test)):
        a = time.process_time()
        model = GaussianNB()
        model.fit(X_train[i], y_train[i])
        predicted_target = model.predict(X_test[i])

        current_acc = accuracy_score(y_test[i], predicted_target)
        current_roc = roc_auc_score(y_test[i], predicted_target)
        b = time.process_time()

        if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
            max_acc = current_acc
            max_time_acc = b - a
            max_seed_acc = i
        
        if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
            max_roc = current_roc
            max_time_roc = b - a
            max_seed_roc = i

    print("Accuracy Gaussian Naive Bayes method :", current_acc, "\nTime to process :", b - a, "seed :", max_seed_acc)
    print("ROC Gaussian Naive Bayes method :", current_roc, "seed :", max_seed_roc)