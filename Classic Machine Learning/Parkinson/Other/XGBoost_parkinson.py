from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import time
from math import inf
from xgboost import XGBClassifier

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

boosters = ["gbtree", "gblinear", "dart"]
eta_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
gamma_ = [0.001, 0.01, 1, 10, 100, 1000]
max_depth_ = [4, 5, 6, 7, 8, 10]
subsample_ = [0.4, 0.5, 0.7, 0.75, 1]

max_acc = 0
max_parameter = (0, 0, 0, 0, 0)
max_time = inf

file = open("parkinsons.data")
data = []
target = []

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

for i in range(len(y_train)):
    y_train[i] = int(y_train[i])

for i in range(len(y_test)):
    y_test[i] = int(y_test[i])

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i in range(len(boosters)):
    print("* computing for booster", boosters[i], "*")
    if boosters[i] != "gblinear":
        for j in range(len(eta_)):
            print("** computing for eta =", eta_[j], "**")
            for k in range(len(gamma_)):
                print("*** computing for gamma =", gamma_[k], "***")
                for l in range(len(max_depth_)):
                    print("**** computing for max_depth =", max_depth_[l], "****")
                    for m in range(len(subsample_)):

                            a = time.process_time()
                            model = XGBClassifier(use_label_encoder=False, disable_default_eval_metric=True, booster=boosters[i], eta=eta_[j], gamma=gamma_[k], max_depth=max_depth_[l], subsample=subsample_[m], sampling_method="uniform")
                            model.fit(X_train, y_train)
                            predicted_target = model.predict(X_test)

                            current_accuracy = accuracy_score(y_test, predicted_target)
                            b = time.process_time()

                            if max_acc < current_accuracy or ( max_acc == current_accuracy and b - a < max_time):
                                max_acc = current_accuracy
                                max_parameter = boosters[i], eta_[j], gamma_[k], max_depth_[l], subsample_[m]
                                max_time = b - a
    else:
        a = time.process_time()
        model = XGBClassifier(use_label_encoder=False, disable_default_eval_metric=True, booster=boosters[i])
        model.fit(X_train, y_train)
        predicted_target = model.predict(X_test)

        current_accuracy = accuracy_score(y_test, predicted_target)
        b = time.process_time()

        if max_acc < current_accuracy or (max_acc == current_accuracy and b - a < max_time):
            max_acc = current_accuracy
            max_parameter = boosters[i]
            max_time = b - a


if max_parameter != "gblinear":
    print("             Best model with SVM :\nbooster =", max_parameter[0], ", eta =", max_parameter[1], ", gamma :", max_parameter[2], ", maximum depth :", max_parameter[3], ", subsample :", max_parameter[4], "\nBest accuracy :", max_acc, "\nTime to process :", max_time)
else:
    print("             Best model with SVM :\nbooster =", max_parameter, "\nBest accuracy :", max_acc, "\nTime to process :", max_time)



