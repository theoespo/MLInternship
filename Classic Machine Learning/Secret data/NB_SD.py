from scipy.sparse.lil import _prepare_index_for_memoryview
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing
import time, random
from math import inf
import numpy as np
from sklearn.naive_bayes import GaussianNB

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
        oh_vector_i = [0 for i in range(max_y)]
        oh_vector_i[y[i] - 1] = 1
        oh_vector.append(oh_vector_i)
    return oh_vector

def n_class(y):
    classes = []
    for i in y:
        if not(i in classes):
            classes.append(i)
    return len(classes)

max_acc = 0
max_roc = 0
max_time_acc = inf
max_time_roc = inf

file = open("MMCR_2021 .csv")


#########################################################################################################################################
# All data (absurd value) (best acc : 75.862%) (best roc : 77.759%)
# with the new absurd value results are better

# data_type = 0
# data = []
# target = []
# for line in file:
#     l = line.split(",")
#     if l[0] != "id":
#         data.append(l[2:-1] + [l[-1][:-1]])
#         target.append(int(l[1]))

# data = num_list2(data)


# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=49, shuffle=True)


# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# y_test_oh = one_hot_encode(y_test, 3)


#########################################################################################################################################
# All data with different random state (absurd value) (best acc : 88.793% with random_state=33)

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
# Sliced data 2 on 6 (absurd value)

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
# sliced data 3 on 6 (absurd value)


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
# sliced data 4 on 6 (absurd value)


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
# sliced data 5 on 6 (absurd value) (best acc : 87.931% with columns=(2, 3, 5, 6, 7), C=1000, gamma=0.01, kernel=rbf) (best roc : 87.698% with columns=(2, 3, 5, 6, 7), C=1000, gamma=0.01, kernel=rbf)
# with the new absurd value results are worse : acc=87.068%, roc=86.44%

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
# All data with different random state (deleted line) (best acc : 85.217% with random_state=29)

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
# for i in range(20, 40):
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
# All data but 2 class with different random state (deleted line) (best acc : 86.086 with random_state=29, C=0.01, gamma=0.0001, kernel=linear) (best roc : 75.615 with random_state=52, C=100, gamma=0.001, kernel=rbf)

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
# for i in range(150, 200):
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
# All data but 2 class with different random state, 5 selected features on 6 (deleted line) (best acc : 85.217 with random_state=19, C=100, gamma=0.0001, kernel=rbf) (best roc : 77.941 with random_state=167, C=100, gamma=1, kernel=rbf)

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
# for i in range(150, 200):
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
# All data but 2 class with different random state (absurd value) (best acc : 87.931% with random_state=33 and 195, C=100, gamma=0.01, kernel=rbf) (best roc : 80.129% with random_state=33, C=1000, gamma=0.01, kernel=rbf)
# with the new absurd value, results are better

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
# for i in range(150, 200):
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
# All data with different random state (average value) (best acc : 87.931% with random_state=33, C=100, gamma=0.01, kernel=rbf)

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
# for i in range(20, 40):
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
    a = time.process_time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    predicted_target = model.predict(X_test)
    predicted_target_vector = one_hot_encode(predicted_target, 3)

    current_acc = accuracy_score(y_test, predicted_target)
    current_roc = roc_auc_score(y_test_oh, predicted_target_vector, multi_class='ovo')
    b = time.process_time()

    print("Accuracy Gaussian Naive Bayes method :", current_acc, "\nTime to process :", b - a)
    print("ROC AUC score Gaussian Naive Bayes method :", current_roc)
if data_type == 2:

    for i in range(len(X_test)):

        if n_class(y_test_oh[i]) != 1:
            a = time.process_time()
            model = GaussianNB()
            model.fit(X_train[i], y_train[i])
            predicted_target = model.predict(X_test[i])
            predicted_target_vector = one_hot_encode(predicted_target, 3)

            current_acc = accuracy_score(y_test[i], predicted_target)
            # current_roc = roc_auc_score(y_test_oh[i], predicted_target_vector, multi_class='ovo')
            b = time.process_time()
            
            if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                max_acc = current_acc
                max_time_acc = b - a
                max_seed = i
            
            # if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
            #     max_roc = current_roc
            #     max_time_roc = b - a

    print("Accuracy Gaussian Naive Bayes method :", max_acc, max_seed)
    # print("ROC AUC score Gaussian Naive Bayes method :", max_roc)
    
if data_type == 1:

    for i in range(len(X_test)):

        if n_class(y_test_oh[i]) != 1:
            a = time.process_time()
            model = GaussianNB()
            model.fit(X_train[i], y_train[i])
            predicted_target = model.predict(X_test[i])
            predicted_target_vector = one_hot_encode(predicted_target, 3)

            current_acc = accuracy_score(y_test[i], predicted_target)
            current_roc = roc_auc_score(y_test_oh[i], predicted_target_vector, multi_class='ovo')
            b = time.process_time()
            
            if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                max_acc = current_acc
                max_time_acc = b - a
                max_col_acc = i
            
            if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                max_roc = current_roc
                max_time_roc = b - a
                max_col_roc = i

    columns_acc = reference[max_col_acc]
    columns_roc = reference[max_col_roc]

    print("Accuracy Gaussian Naive Bayes method :", max_acc, columns_acc)
    print("ROC AUC score Gaussian Naive Bayes method :", max_roc)