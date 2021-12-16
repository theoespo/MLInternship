from numpy.lib.function_base import average
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, tree
from sklearn.feature_selection import RFE
import time
from math import inf

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

mean_len = 500
MSS = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
criterion_ = ["entropy", "gini"]
MSS_time = [0 for i in range(len(MSS))]

max_acc = [0, 0]
max_parameter = [(0, 0), (0, 0)]
max_time = [0, 0]

mean_max = 0
all_max_parameter = []
weight_max_parameter = []
average_max_accuracy = []

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

for j in range(mean_len):

    print("* computing", j+1, "/", mean_len, "*")
    max_acc[0] = 0
    max_time[0] = inf
    for i in range(len(MSS)):

        for k in range(2):

            a = time.process_time()
            estimator = tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[k])
            model = RFE(estimator, n_features_to_select=5)
            model.fit(X_train, y_train)
            predicted_target = model.predict(X_test)

            current_acc = truncate(accuracy_score(y_test, predicted_target), 5)
            b = time.process_time()
            MSS_time[i] += b - a

            if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time[0]):
                max_acc[0] = current_acc
                max_parameter[0] = (MSS[i], criterion_[k])
                max_time[0] = b - a
            if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time[1]):
                max_acc[1] = current_acc
                max_parameter[1] = (MSS[i], criterion_[k])
                max_time[1] = b - a

    
    mean_max += max_acc[0]
    if not(max_parameter[0] in all_max_parameter):
        all_max_parameter.append(max_parameter[0])
        weight_max_parameter.append(1)
        average_max_accuracy.append(max_acc[0])
    else:
        weight_max_parameter[all_max_parameter.index(max_parameter[0])] += 1
        average_max_accuracy[all_max_parameter.index(max_parameter[0])] += max_acc[0]

mean_max = mean_max/mean_len
for i in range(len(MSS_time)):
    MSS_time[i] = MSS_time[i]/(len(MSS)*len(criterion_)*mean_len)
for i in range(len(average_max_accuracy)):
    average_max_accuracy[i] = truncate(average_max_accuracy[i]/weight_max_parameter[i], 5)
for i in range(len(weight_max_parameter)):
    weight_max_parameter[i] = weight_max_parameter[i]/(mean_len/100)

print("\n* Average of maximum accuracy :", mean_max, "\n* Parameters giving a maximum of accuracy :", all_max_parameter, "\n--> with a percentage of", weight_max_parameter, "\n--> with an average accuracy of :", average_max_accuracy)
print("\n===================================================================================================================================\n")
print("Average time for different value of minimum samples split :\n", MSS_time)
print("\n===================================================================================================================================\n")
print("          Best model : \n\nMinimum samples split =", max_parameter[1][0], ", criterion :", max_parameter[1][1],"\nAccuracy :", max_acc[1], "\nTime to process :", max_time[1])

