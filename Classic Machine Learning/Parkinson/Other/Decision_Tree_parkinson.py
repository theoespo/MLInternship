from numpy.lib.function_base import average
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing, tree
import time, os, pickle
from pathlib import Path
from math import inf

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(float(data[i]))
    return data_num

mean_len = 500
MSS = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
criterion_ = ["entropy", "gini"]
MSS_time = [0 for i in range(len(MSS))]

filename = os.path.basename(__file__)[:-13]
path_model = Path("model")
path_perf = Path("perf")

f = open(path_perf / ("perf_" + filename + "_best_model.txt"), "a")
f.close()
f = open(path_perf / ("perf_" + filename + "_best_model.txt"))
    
prev_perf = []
for line in f:
    l = line.split(",")
    for i in range(len(l)):
        prev_perf += [float(l[i])]
f.close()

if len(prev_perf) != 0:
    max_acc = [0, prev_perf[0]]
    max_time_acc = [inf, prev_perf[1]]
else:
    max_acc = [0, 0]
    max_time_acc = [inf, inf]

max_parameter_acc = [(0, 0), (0, 0)]
max_roc = 0
max_parameter_roc = (0, 0)
max_time_roc = inf

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
file.close()

X_train, X_test, y_train, y_test = train_test_split(data[1:], target[1:], test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for j in range(mean_len):

    print("* computing", j, "/", mean_len, "*")
    max_acc[0] = 0
    max_time_acc[0] = inf
    for i in range(len(MSS)):

        for k in range(2):


            a = time.process_time()
            model = tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[k])
            model.fit(X_train, y_train)
            predicted_target = model.predict(X_test)

            current_acc = accuracy_score(y_test, predicted_target)
            current_roc = roc_auc_score(num_list(y_test), num_list(predicted_target))
            b = time.process_time()
            MSS_time[i] += b - a

            if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                max_acc[0] = current_acc
                max_parameter_acc[0] = (MSS[i], criterion_[k])
                max_time_acc[0] = b - a
                
            if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                max_acc[1] = current_acc
                max_parameter_acc[1] = (MSS[i], criterion_[k])
                max_time_acc[1] = b - a
                pickle.dump(model, open(path_model / (filename  + "_Best_Model.sav"), "wb"))
                f = open(path_perf / ("perf_" + filename + "_best_model.txt"), "w")
                f.write(str(max_acc[1]) + ", " + str(max_time_acc[1]))
                f.close()

            if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                max_roc = current_roc
                max_parameter_roc = (MSS[i], criterion_[k])
                max_time_roc = b - a
            

    
    mean_max += max_acc[0]
    if not(max_parameter_acc[0] in all_max_parameter):
        all_max_parameter.append(max_parameter_acc[0])
        weight_max_parameter.append(1)
        average_max_accuracy.append(max_acc[0])
    else:
        weight_max_parameter[all_max_parameter.index(max_parameter_acc[0])] += 1
        average_max_accuracy[all_max_parameter.index(max_parameter_acc[0])] += max_acc[0]

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
print("          Best model DT (accuracy) : \n\nMinimum samples split =", max_parameter_acc[1][0], ", criterion :", max_parameter_acc[1][1],"\nAccuracy :", truncate(max_acc[1], 5), "\nTime to process :", max_time_acc[1])
print("\n===================================================================================================================================\n")
print("          Best model DT (ROC) : \n\nMinimum samples split =", max_parameter_roc[0], ", criterion :", max_parameter_roc[1],"\nROC AUC :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

