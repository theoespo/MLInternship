from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import preprocessing, svm
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

C_ =  [0.0001, 0.001, 0.01, 1, 10, 100, 1000]
gamma_ = [0.0001, 0.001, 0.01, 1, 10, 100, 1000]
kernel_ = ['linear', 'rbf', 'sigmoid']

max_acc = 0
max_roc = 0
max_parameter_acc = (-1, -1, -1)
max_parameter_roc = (-1, -1, -1)
max_col_acc = -1
max_col_roc = -1
max_time_acc = inf
max_time_roc = inf


file = open("MMCR_2021 .csv")


#########################################################################################################################################
# All data (absurd value) (best acc : 88.793% with C=100, gamma=0.01, kernel=rbf) (best roc : 86.54% with C=1000, gamma=0.01, kernel=rbf)
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


# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33, shuffle=True)


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
# for i in range(100, 150):

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
    for i in range(len(C_)):

        print("C =", C_[i])
        for j in range(len(gamma_)):

            for k in range(len(kernel_)):
                
                a = time.process_time()
                model = svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j])
                model.fit(X_train, y_train)
                predicted_target = model.predict(X_test)
                b = time.process_time()

                predicted_target_vector = one_hot_encode(predicted_target, 3)          

                current_acc = accuracy_score(y_test, predicted_target)
                current_roc = roc_auc_score(y_test_oh, predicted_target_vector, multi_class='ovo')

                if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                    max_acc = current_acc
                    max_parameter_acc = (C_[i], gamma_[j], kernel_[k])
                    max_time_acc = b - a
                
                if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                    max_roc = current_roc
                    max_parameter_roc = (C_[i], gamma_[j], kernel_[k])
                    max_time_roc = b - a
        
    print("\n===============================================================================================\n")
    print("             Best model with SVM (accuracy) :\nC =", max_parameter_acc[0], ", gamma =", max_parameter_acc[1], ", kernel :", max_parameter_acc[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc)
    print("\n===============================================================================================\n")
    print("             Best model with SVM (ROC AUC) :\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc)

elif data_type == 1:
    for i in range(len(C_)):

        print("**** C =", C_[i])
        for j in range(len(gamma_)):

            print("°° gamma =", gamma_[j])
            for k in range(len(kernel_)):

                print("$ kernel =", kernel_[k])
                for l in range(len(X_train)):
                    
                    a = time.process_time()
                    model = svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()

                    predicted_target_vector = one_hot_encode(predicted_target, 3)          

                    current_acc = accuracy_score(y_test[l], predicted_target)
                    current_roc = roc_auc_score(y_test_oh[l], predicted_target_vector, multi_class='ovo')

                    if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                        max_acc = current_acc
                        max_parameter_acc = (C_[i], gamma_[j], kernel_[k])
                        max_col_acc = l
                        max_time_acc = b - a
                    
                    if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                        max_roc = current_roc
                        max_parameter_roc = (C_[i], gamma_[j], kernel_[k])
                        max_col_roc = l
                        max_time_roc = b - a


    columns_acc = reference[max_col_acc]
    columns_roc = reference[max_col_roc]


    print("\n===============================================================================================\n")
    print("             Best model with SVM (accuracy) :\nC =", max_parameter_acc[0], ", gamma =", max_parameter_acc[1], ", kernel :", max_parameter_acc[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc, "\nColumns :", columns_acc)
    print("\n===============================================================================================\n")
    print("             Best model with SVM (ROC AUC) :\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc, "\nColumns :", columns_roc)

elif data_type == 2:
    for i in range(len(C_)):

        print("**** C =", C_[i])
        for j in range(len(gamma_)):

            print("°° gamma =", gamma_[j])
            for k in range(len(kernel_)):

                print("$ kernel =", kernel_[k])
                for l in range(len(X_train)):
                    
                    a = time.process_time()
                    model = svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()

                    predicted_target_vector = one_hot_encode(predicted_target, 3)          

                    current_acc = accuracy_score(y_test[l], predicted_target)

                    # if n_class(y_test[l]) > 1 and n_class(predicted_target) > 1:
                    #     current_roc = roc_auc_score(y_test_oh[l], predicted_target_vector, multi_class='ovo')

                    if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                        max_acc = current_acc
                        max_parameter_acc = (C_[i], gamma_[j], kernel_[k])
                        random_state_acc = l
                        max_time_acc = b - a
                    
                    # if n_class(y_test[l]) > 1 and n_class(predicted_target) > 1:
                    #     if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                    #         max_roc = current_roc
                    #         max_parameter_roc = (C_[i], gamma_[j], kernel_[k])
                    #         random_state_roc = l
                    #         max_time_roc = b - a


    print("\n===============================================================================================\n")
    print("             Best model with SVM (accuracy) :\nC =", max_parameter_acc[0], ", gamma =", max_parameter_acc[1], ", kernel :", max_parameter_acc[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc, "\nRandom state :", random_state_acc)
    print("\n===============================================================================================\n")
    # print("             Best model with SVM (ROC AUC) :\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc, "\nRandom state :", random_state_roc)

else:
    for i in range(len(C_)):

        print("**** C =", C_[i])
        for j in range(len(gamma_)):

            print("°° gamma =", gamma_[j])
            for k in range(len(kernel_)):

                print("$ kernel =", kernel_[k])
                for l in range(len(X_train)):
                    
                    a = time.process_time()
                    model = svm.SVC(kernel=kernel_[k], C=C_[i], gamma=gamma_[j])
                    model.fit(X_train[l], y_train[l])
                    predicted_target = model.predict(X_test[l])
                    b = time.process_time()

                    current_acc = accuracy_score(y_test[l], predicted_target)

                    if n_class(y_test[l]) > 1 and n_class(predicted_target) > 1:
                        current_roc = roc_auc_score(y_test[l], predicted_target)

                    if (max_acc < current_acc) or (max_acc == current_acc and b - a < max_time_acc): 
                        max_acc = current_acc
                        max_parameter_acc = (C_[i], gamma_[j], kernel_[k])
                        random_state_acc = l
                        max_time_acc = b - a
                    
                    if n_class(y_test[l]) > 1 and n_class(predicted_target) > 1:
                        if max_roc < current_roc or (max_roc == current_roc and b - a < max_time_roc):
                            max_roc = current_roc
                            max_parameter_roc = (C_[i], gamma_[j], kernel_[k])
                            random_state_roc = l
                            max_time_roc = b - a


    print("\n===============================================================================================\n")
    print("             Best model with SVM (accuracy) :\nC =", max_parameter_acc[0], ", gamma =", max_parameter_acc[1], ", kernel :", max_parameter_acc[2], "\nBest accuracy :", truncate(max_acc, 5), "\nTime to process :", max_time_acc, "\nRandom state :", random_state_acc)
    print("\n===============================================================================================\n")
    print("             Best model with SVM (ROC AUC) :\nC =", max_parameter_roc[0], ", gamma =", max_parameter_roc[1], ", kernel :", max_parameter_roc[2], "\nBest ROC AUC value :", truncate(max_roc, 5), "\nTime to process :", max_time_roc, "\nRandom state :", random_state_roc)

