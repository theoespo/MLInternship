from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing, tree
import time, random
from math import inf
import numpy as np

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
            if data[i][j] != '':
                data_num[i].append(float(data[i][j]))
            else:
                data_num[i].append(-1000)
    return data_num

def one_hot_encode(y, max_y):
    oh_vector = []
    for i in range(len(y)):
        oh_vector_i = np.zeros(shape=(max_y,))
        oh_vector_i[y[i] - 1] = 1
        oh_vector.append(oh_vector_i)
    return oh_vector

def n_class(y):
    classes = []
    for i in y:
        if not(i in classes):
            classes.append(i)
    return len(classes)

mean_len = 100
MSS = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
criterion_ = ["entropy", "gini"]
MSS_time = [0 for i in range(len(MSS))]

max_parameter_acc = [(0, 0), (0, 0)]
max_roc = 0
max_parameter_roc = (0, 0)
max_time_roc = inf
max_acc = [0, 0]
max_time_acc = [inf, inf]

mean_max = 0
all_max_parameter = []
weight_max_parameter = []
average_max_accuracy = []


file = open("MMCR_2021 .csv")


#########################################################################################################################################
# All data (absurd value) (best acc : 77.586% with MSS=25, criterion=entropy) (best roc : 77.964% with MSS=2, criterion=gini)

# data_type = 0
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         data.append(l[2:-1] + [l[-1][:-1]])
#         target.append(int(l[1]))

# data = num_list2(data)


# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33, shuffle=True)


# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# y_test_oh = one_hot_encode(y_test, 3)


#########################################################################################################################################
# All data with different random state (absurd value) (best acc : 84.482% with random_state=44, MSS=50, criterion=gini)

# data_type = 2
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         data.append(l[2:-1] + [l[-1][:-1]])
#         target.append(int(l[1]))

# data = num_list2(data)


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(50, 200):

#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)
#     y_test_oh_small = one_hot_encode(y_test_small, 3)

#     X_train.append(X_train_small)
#     X_test.append(X_test_small)
#     y_train.append(y_train_small)
#     y_test_oh.append(y_test_oh_small)
#     y_test.append(y_test_small)
    


#########################################################################################################################################
# Sliced data 2 on 6 (absurd value) (best acc : 84.482% with MSS=50, criterion=entropy, columns=(5,6)) (best roc : 80.893% with MSS=50, criterion=entropy, columns=(5, 6))

# data_type = 1
# data = dict()
# target = dict()
# reference = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         if i < j:
#             reference.append((i, j))
#             data[(i, j)] = []
#             target[(i, j)] = []

# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         for i in range(2, 8):
#             for j in range(2, 8):
#                 if i < j:
#                     if l[i] == '':
#                         data[(i, j)].append([-1000] + [float(l[j])])
#                     elif l[j] == '':
#                         data[(i, j)].append([float(l[i])] + [-1000])
#                     else:
#                         data[(i, j)].append([float(l[i])] + [float(l[j])])
#                     target[(i, j)].append(int(l[1]))


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         if i < j:
            
#             X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data[(i, j)], target[(i, j)], test_size=0.25, random_state=33, shuffle=True)

#             scaler = preprocessing.StandardScaler().fit(X_train_small)
#             X_train_small = scaler.transform(X_train_small)
#             X_test_small = scaler.transform(X_test_small)
#             y_test_oh_small = one_hot_encode(y_test_small, 3)

#             X_train.append(X_train_small)
#             X_test.append(X_test_small)
#             y_train.append(y_train_small)
#             y_test_oh.append(y_test_oh_small)
#             y_test.append(y_test_small)
    
#########################################################################################################################################
# sliced data 3 on 6 (absurd value) (best acc : 82.758% with MSS=50, criterion=gini, columns=(2, 3, 5)) (best roc : 84.578% with MSS=50, criterion=gini, columns=(2, 3, 5))


# data_type = 1
# data = dict()
# target = dict()
# reference = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             if i < j and j < k:
#                 reference.append((i, j, k))
#                 data[(i, j, k)] = []
#                 target[(i, j, k)] = []

# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         for i in range(2, 8):
#             for j in range(2, 8):
#                 for k in range(2, 8):
#                     if i < j and j < k:
#                         if l[i] == '':
#                             data[(i, j, k)].append([-1] + [float(l[j])] + [float(l[k])])
#                         elif l[j] == '':
#                             data[(i, j, k)].append([float(l[i])] + [-1] + [float(l[k])])
#                         elif l[k] == '':
#                             data[(i, j, k)].append([float(l[i])] + [float(l[j])] + [-1])
#                         else:
#                             data[(i, j, k)].append([float(l[i])] + [float(l[j])] + [float(l[k])])
#                         target[(i, j, k)].append(int(l[1]))


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             if i < j and j < k:
                
#                 X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data[(i, j, k)], target[(i, j, k)], test_size=0.25, random_state=33, shuffle=True)

#                 scaler = preprocessing.StandardScaler().fit(X_train_small)
#                 X_train_small = scaler.transform(X_train_small)
#                 X_test_small = scaler.transform(X_test_small)
#                 y_test_oh_small = one_hot_encode(y_test_small, 3)

#                 X_train.append(X_train_small)
#                 X_test.append(X_test_small)
#                 y_train.append(y_train_small)
#                 y_test_oh.append(y_test_oh_small)
#                 y_test.append(y_test_small)

#########################################################################################################################################
# sliced data 4 on 6 (absurd value) (best acc : 81.034% with MSS=70, criterion=entropy, columns=(4, 5, 6, 7)) (best roc : 84.86% with MSS=50, criterion=gini, columns=(2, 3, 5, 7))


# data_type = 1
# data = dict()
# target = dict()
# reference = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             for m in range(2, 8):
#                 if i < j and j < k and k < m:
#                     reference.append((i, j, k, m))
#                     data[(i, j, k, m)] = []
#                     target[(i, j, k, m)] = []


# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         for i in range(2, 8):
#             for j in range(2, 8):
#                 for k in range(2, 8):
#                     for m in range(2, 8):
#                         if i < j and j < k and k < m:
#                             if l[i] == '':
#                                 data[(i, j, k, m)].append([-1] + [float(l[j])] + [float(l[k])] + [float(l[m])])
#                             elif l[j] == '':
#                                 data[(i, j, k, m)].append([float(l[i])] + [-1] + [float(l[k])] + [float(l[m])])
#                             elif l[k] == '':
#                                 data[(i, j, k, m)].append([float(l[i])] + [float(l[j])] + [-1] + [float(l[m])])
#                             elif l[m] == '':
#                                 data[(i, j, k, m)].append([float(l[i])] + [float(l[j])] + [float(l[k])] + [-1])
#                             else:
#                                 data[(i, j, k, m)].append([float(l[i])] + [float(l[j])] + [float(l[k])] + [float(l[m])])
#                             target[(i, j, k, m)].append(int(l[1]))


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             for m in range(2, 8):
#                 if i < j and j < k and k < m:
                    
#                     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data[(i, j, k, m)], target[(i, j, k, m)], test_size=0.25, random_state=33, shuffle=True)

#                     scaler = preprocessing.StandardScaler().fit(X_train_small)
#                     X_train_small = scaler.transform(X_train_small)
#                     X_test_small = scaler.transform(X_test_small)
#                     y_test_oh_small = one_hot_encode(y_test_small, 3)

#                     X_train.append(X_train_small)
#                     X_test.append(X_test_small)
#                     y_train.append(y_train_small)
#                     y_test_oh.append(y_test_oh_small)
#                     y_test.append(y_test_small)

#########################################################################################################################################
# sliced data 5 on 6 (absurd value) (best acc : 78.448% with MSS=25, criterion=entropy, columns=(2, 4, 5, 6, 7)) (best roc : 87.698% with MSS=2, criterion=gini, columns=(3, 4, 5, 6, 7))
# ROC between 84% and 87.6%

# data_type = 1
# data = dict()
# target = dict()
# reference = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             for m in range(2, 8):
#                 for n in range(2, 8):
#                     if i < j and j < k and k < m and m < n:
#                         reference.append((i, j, k, m, n))
#                         data[(i, j, k, m, n)] = []
#                         target[(i, j, k, m, n)] = []


# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         for i in range(2, 8):
#             for j in range(2, 8):
#                 for k in range(2, 8):
#                     for m in range(2, 8):
#                         for n in range(2, 8):
#                             if i < j and j < k and k < m and m < n:
#                                 if l[i] == '':
#                                     data[(i, j, k, m, n)].append([-1000] + [float(l[j])] + [float(l[k])] + [float(l[m])] + [float(l[n])])
#                                 elif l[j] == '':
#                                     data[(i, j, k, m, n)].append([float(l[i])] + [-1000] + [float(l[k])] + [float(l[m])] + [float(l[n])])
#                                 elif l[k] == '':
#                                     data[(i, j, k, m, n)].append([float(l[i])] + [float(l[j])] + [-1000] + [float(l[m])] + [float(l[n])])
#                                 elif l[m] == '':
#                                     data[(i, j, k, m, n)].append([float(l[i])] + [float(l[j])] + [float(l[k])] + [-1000] + [float(l[n])])
#                                 elif l[n] == '':
#                                     data[(i, j, k, m, n)].append([float(l[i])] + [float(l[j])] + [float(l[k])] + [float(l[m])] + [-1000])
#                                 else:
#                                     data[(i, j, k, m, n)].append([float(l[i])] + [float(l[j])] + [float(l[k])] + [float(l[m])] + [float(l[n])])
#                                 target[(i, j, k, m, n)].append(int(l[1]))


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(2, 8):
#     for j in range(2, 8):
#         for k in range(2, 8):
#             for m in range(2, 8):
#                 for n in range(2, 8):
#                     if i < j and j < k and k < m and m < n:
                        
#                         X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data[(i, j, k, m, n)], target[(i, j, k, m, n)], test_size=0.25, random_state=33, shuffle=True)

#                         scaler = preprocessing.StandardScaler().fit(X_train_small)
#                         X_train_small = scaler.transform(X_train_small)
#                         X_test_small = scaler.transform(X_test_small)
#                         y_test_oh_small = one_hot_encode(y_test_small, 3)

#                         X_train.append(X_train_small)
#                         X_test.append(X_test_small)
#                         y_train.append(y_train_small)
#                         y_test_oh.append(y_test_oh_small)
#                         y_test.append(y_test_small)


#########################################################################################################################################
# All data with different random state (deleted line) (best acc : 82.608% with MSS=50, criterion=gini, random_state=84)

# data_type = 2
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id" and l[-2] != '':
#         data.append(l[2:-1] + [l[-1][:-1]])
#         target.append(int(l[1]))

# data = num_list2(data)


# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(0, 200):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)
#     y_test_oh_small = one_hot_encode(y_test_small, 3)

#     if n_class(y_test_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test_oh.append(y_test_oh_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data but 2 class with different random state (deleted line) (best acc : 83.478% with random_state=49, MSS=50, criterion=entropy) (best roc : 78.366% with random_state=45, MSS=5, criterion=gini)
# Roc can be less : 78.26%

# data_type = 3
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id" and l[-2] != '':
#         data.append(l[2:-1] + [l[-1][:-1]])
#         if l[1] != '3':
#             target.append(int(l[1]))
#         else:
#             target.append(1)

# data = num_list2(data)


# X_train = []
# X_test = []
# y_train = []
# y_test = []
# for i in range(0, 200):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)

#     if n_class(y_train_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data but 2 class with different random state, 5 selected features on 6 (deleted line) (best acc : 84.347% with random_state=173, MSS=10, criterion=gini) (best roc : 80.934% with random_state=173, MSS=10, criterion=gini)

# data_type = 3
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id" and l[-2] != '':
#         data.append(l[2:4] + l[5:-1]+ [l[-1][:-1]])
#         if l[1] != '3':
#             target.append(int(l[1]))
#         else:
#             target.append(1)

# data = num_list2(data)


# X_train = []
# X_test = []
# y_train = []
# y_test = []
# for i in range(0, 200):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)

#     if n_class(y_train_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data but 2 class with different random state (absurd value) (best acc : 83.62% with random_state=195, MSS=70, criterion=gini) (best roc : 76.436% with random_state=97, MSS=15, criterion=gini)
# ROC can be worse 75.862%

# data_type = 3
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         data.append(l[2:-1] + [l[-1][:-1]])
#         if l[1] != '3':
#             target.append(int(l[1]))
#         else:
#             target.append(1)

# data = num_list2(data)


# X_train = []
# X_test = []
# y_train = []
# y_test = []
# for i in range(0, 200):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)

#     if n_class(y_train_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data with different random state (average value) (best acc : 83.62% with random_state=62, MSS=60, criterion=entropy)

# data_type = 2
# data = []
# target = []
# times = 0
# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         data.append(l[2:-1] + [l[-1][:-1]])
#         target.append(int(l[1]))
#         if l[6] != '':
#             times += float(l[6])

# times = times/(len(data) - 2)
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         if data[i][j] == '':
#             data[i][j] = times
# data = num_list2(data)

# X_train = []
# X_test = []
# y_train = []
# y_test_oh = []
# y_test = []
# for i in range(0, 200):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)
#     y_test_oh_small = one_hot_encode(y_test_small, 3)

#     X_train.append(X_train_small)
#     X_test.append(X_test_small)
#     y_train.append(y_train_small)
#     y_test_oh.append(y_test_oh_small)
#     y_test.append(y_test_small)

#########################################################################################################################################

#########################################################################################################################################

if data_type == 0:
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
                b = time.process_time()

                predicted_target_vector = one_hot_encode(predicted_target, 3)          

                current_acc = accuracy_score(y_test, predicted_target)
                current_roc = roc_auc_score(y_test_oh, predicted_target_vector, multi_class='ovo')

                MSS_time[i] += b - a

                if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                    max_acc[0] = current_acc
                    max_parameter_acc[0] = (MSS[i], criterion_[k])
                    max_time_acc[0] = b - a
                    
                if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                    max_acc[1] = current_acc
                    max_parameter_acc[1] = (MSS[i], criterion_[k])
                    max_time_acc[1] = b - a

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


elif data_type == 1:

    for j in range(mean_len):

        print("* computing", j, "/", mean_len, "*")
        max_acc[0] = 0
        max_time_acc[0] = inf
        for i in range(len(MSS)):

            for k in range(2):

                for l in range(len(X_train)):

                    a = time.process_time()
                    model = tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[k])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()

                    predicted_target_vector = one_hot_encode(predicted_target, 3)          

                    current_acc = accuracy_score(y_test[l], predicted_target)
                    current_roc = roc_auc_score(y_test_oh[l], predicted_target_vector, multi_class='ovo')
                    MSS_time[i] += b - a

                    if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                        max_acc[0] = current_acc
                        max_parameter_acc[0] = (MSS[i], criterion_[k])
                        max_time_acc[0] = b - a
                        
                    if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                        max_acc[1] = current_acc
                        max_parameter_acc[1] = (MSS[i], criterion_[k])
                        max_time_acc[1] = b - a
                        max_col_acc = l

                    if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                        max_roc = current_roc
                        max_parameter_roc = (MSS[i], criterion_[k])
                        max_time_roc = b - a
                        max_col_roc = l
                

        
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

    columns_acc = reference[max_col_acc]
    columns_roc = reference[max_col_roc]

    print("\n* Average of maximum accuracy :", mean_max, "\n* Parameters giving a maximum of accuracy :", all_max_parameter, "\n--> with a percentage of", weight_max_parameter, "\n--> with an average accuracy of :", average_max_accuracy)
    print("\n===================================================================================================================================\n")
    print("Average time for different value of minimum samples split :\n", MSS_time)
    print("\n===================================================================================================================================\n")
    print("          Best model DT (accuracy) : \n\nMinimum samples split =", max_parameter_acc[1][0], ", criterion :", max_parameter_acc[1][1], ", columns :", columns_acc, "\nAccuracy :", truncate(max_acc[1], 5), "\nTime to process :", max_time_acc[1])
    print("\n===================================================================================================================================\n")
    print("          Best model DT (ROC) : \n\nMinimum samples split =", max_parameter_roc[0], ", criterion :", max_parameter_roc[1], ", columns :", columns_roc, "\nROC AUC :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

elif data_type == 2:

    for j in range(mean_len):

        print("* computing", j, "/", mean_len, "*")
        max_acc[0] = 0
        max_time_acc[0] = inf
        for i in range(len(MSS)):

            for k in range(2):

                for l in range(len(X_train)):

                    a = time.process_time()
                    model = tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[k])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()

                    predicted_target_vector = one_hot_encode(predicted_target, 3)

                    current_acc = accuracy_score(y_test[l], predicted_target)
                    # current_roc = roc_auc_score(y_test_oh[l], predicted_target_vector, multi_class='ovo')
                    MSS_time[i] += b - a

                    if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                        max_acc[0] = current_acc
                        max_parameter_acc[0] = (MSS[i], criterion_[k])
                        max_time_acc[0] = b - a
                        
                    if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                        max_acc[1] = current_acc
                        max_parameter_acc[1] = (MSS[i], criterion_[k])
                        max_time_acc[1] = b - a
                        random_state_acc = l

                    # if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                    #     max_roc = current_roc
                    #     max_parameter_roc = (MSS[i], criterion_[k])
                    #     max_time_roc = b - a
                    #     random_state_roc = l
                

        
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
    print("          Best model DT (accuracy) : \n\nMinimum samples split =", max_parameter_acc[1][0], ", criterion :", max_parameter_acc[1][1], ", random_state=", random_state_acc, "\nAccuracy :", truncate(max_acc[1], 5), "\nTime to process :", max_time_acc[1])
    print("\n===================================================================================================================================\n")
    # print("          Best model DT (ROC) : \n\nMinimum samples split =", max_parameter_roc[0], ", criterion :", max_parameter_roc[1], ", random_state=", random_state_roc, "\nROC AUC :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

else:

    for j in range(mean_len):

        print("* computing", j, "/", mean_len, "*")
        max_acc[0] = 0
        max_time_acc[0] = inf
        for i in range(len(MSS)):

            for k in range(2):

                for l in range(len(X_train)):

                    a = time.process_time()
                    model = tree.DecisionTreeClassifier(min_samples_split=MSS[i], criterion=criterion_[k])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()
            
                    predicted_target_vector = one_hot_encode(predicted_target, 3)

                    current_acc = accuracy_score(y_test[l], predicted_target)
                    current_roc = roc_auc_score(y_test[l], predicted_target)
                    MSS_time[i] += b - a

                    if max_acc[0] < current_acc or (max_acc[0] == current_acc and b - a < max_time_acc[0]):
                        max_acc[0] = current_acc
                        max_parameter_acc[0] = (MSS[i], criterion_[k])
                        max_time_acc[0] = b - a
                        
                    if max_acc[1] < current_acc or (max_acc[1] == current_acc and b - a < max_time_acc[1]):
                        max_acc[1] = current_acc
                        max_parameter_acc[1] = (MSS[i], criterion_[k])
                        max_time_acc[1] = b - a
                        random_state_acc = l

                    if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                        max_roc = current_roc
                        max_parameter_roc = (MSS[i], criterion_[k])
                        max_time_roc = b - a
                        random_state_roc = l

        
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
    print("          Best model DT (accuracy) : \n\nMinimum samples split =", max_parameter_acc[1][0], ", criterion :", max_parameter_acc[1][1], ", random_state=", random_state_acc, "\nAccuracy :", truncate(max_acc[1], 5), "\nTime to process :", max_time_acc[1])
    print("\n===================================================================================================================================\n")
    print("          Best model DT (ROC) : \n\nMinimum samples split =", max_parameter_roc[0], ", criterion :", max_parameter_roc[1], ", random_state=", random_state_roc, "\nROC AUC :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)


