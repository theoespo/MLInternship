from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time, random
import numpy as np
import matplotlib.pyplot as plt
from math import inf
from tensorflow.keras import Sequential, layers, metrics, backend, callbacks

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

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        results[i, sequences[i] - 1] = 1.
    return np.array(results)

len_mean = 1
n_epochs = 60
len_layers = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150]
activations = ["selu", "elu"]
optimizers = ["rmsprop"]
# batch_size = [5, 10, 15, 20]

# len_layers = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
# activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
# optimizers = ["rmsprop", "SGD", "adam", "adadelta", "Adagrad", "adamax", "nadam", "Ftrl"]
# batch_size = [5, 10, 15, 20, 30, 40, 50, 80, 100, 150]

max_acc = 0
min_time_acc = inf
min_loss = inf
min_time_loss = inf
max_auc = 0
min_time_auc = inf

max_parameter_acc = -1
max_parameter_loss = -1
max_parameter_auc = -1

# seed = random.randint(0, 2000)
seed = 33
file = open("MMCR_2021 .csv")


#########################################################################################################################################
# All data (absurd value) (best acc : 82.758% with Number of estimators = 90, Minimum samples split = 30 , criterion : entropy) (best roc : 84.205% with Number of estimators = 80, Minimum samples split = 30 , criterion : gini)

    data_type = 0
    data = []
    target = []
    for line in file:
        l = line.split(",")
        if l[0] != "id":
            data.append(l[2:-1] + [l[-1][:-1]])
            target.append(int(l[1]))

    data = np.array(num_list2(data))
    target = np.array(target)


    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed, shuffle=True)


    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = vectorize_sequences(y_train, 3)
    y_test = vectorize_sequences(y_test, 3)


#########################################################################################################################################
# All data with different random state (absurd value) (best acc : 85.344% with Number of estimators = 100, Minimum samples split = 20 , criterion : gini , random_state : 97)

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
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

#     X_train.append(X_train_small)
#     X_test.append(X_test_small)
#     y_train.append(y_train_small)
#     y_test_oh.append(y_test_oh_small)
#     y_test.append(y_test_small)
    


#########################################################################################################################################
# Sliced data 2 on 6 (absurd value) (best acc : 83.62% with Number of estimators = 70, Minimum samples split = 40 , criterion : entropy , columns : (3, 5)) (best roc : 84.86% with Number of estimators = 90, Minimum samples split = 40 , criterion : entropy , columns : (3, 5))

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
#             y_train_small = vectorize_sequences(y_train_small, 3)
#             y_test_small = vectorize_sequences(y_test_small, 3)

#             X_train.append(X_train_small)
#             X_test.append(X_test_small)
#             y_train.append(y_train_small)
#             y_test_oh.append(y_test_oh_small)
#             y_test.append(y_test_small)
    
#########################################################################################################################################
# sliced data 3 on 6 (absurd value) (best acc : 82.758% with Number of estimators = 50, Minimum samples split = 30 , criterion : gini , columns : (2, 5, 6)) (best roc : 85.091% with Number of estimators = 40, Minimum samples split = 50 , criterion : gini , columns : (3, 5, 7))


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
#                 y_train_small = vectorize_sequences(y_train_small, 3)
#                 y_test_small = vectorize_sequences(y_test_small, 3)

#                 X_train.append(X_train_small)
#                 X_test.append(X_test_small)
#                 y_train.append(y_train_small)
#                 y_test_oh.append(y_test_oh_small)
#                 y_test.append(y_test_small)

#########################################################################################################################################
# sliced data 4 on 6 (absurd value) (best acc : 81.034% with MSS=70, criterion=entropy, columns=(4, 5, 6, 7)) (best roc : 85.232% with MSS=50, criterion=gini, columns=(3, 4, 5, 6))


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
#                     y_train_small = vectorize_sequences(y_train_small, 3)
#                     y_test_small = vectorize_sequences(y_test_small, 3)

#                     X_train.append(X_train_small)
#                     X_test.append(X_test_small)
#                     y_train.append(y_train_small)
#                     y_test_oh.append(y_test_oh_small)
#                     y_test.append(y_test_small)

#########################################################################################################################################
# sliced data 5 on 6 (absurd value) Best parameters for minimum loss ( 0.4115052819252014 ) : [40, 'selu', 'elu', 'rmsprop', (2, 3, 4, 5, 7)], Best parameters for maximum accuracy ( 0.8879310488700867 ): [40, 'selu', 'elu', 'rmsprop', (3, 4, 5, 6, 7)], Best parameter for max auc ( 0.9537380933761597 ): [40, 'selu', 'elu', 'rmsprop', (2, 3, 5, 6, 7)]

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
#                         y_train_small = vectorize_sequences(y_train_small, 3)
#                         y_test_small = vectorize_sequences(y_test_small, 3)

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
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

#     if n_class(y_test_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test_oh.append(y_test_oh_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data but 2 class with different random state (deleted line) (best acc : 81.739 with random_state=197, MSS=60, criterion=entropy) (best roc : 73.823% with random_state=167, MSS=2, criterion=gini)

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
# for i in range(0, 150):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

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
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

#     if n_class(y_train_small) > 1:
#         X_train.append(X_train_small)
#         X_test.append(X_test_small)
#         y_train.append(y_train_small)
#         y_test.append(y_test_small)

#########################################################################################################################################
# All data but 2 class with different random state (absurd value) (best acc : 86.086 with random_state=29, C=0.01, gamma=0.0001, kernel=linear) (best roc : 75.615 with random_state=52, C=100, gamma=0.001, kernel=rbf)

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
# for i in range(0, 50):
#     X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(data, target, test_size=0.25, random_state=i, shuffle=True)

#     scaler = preprocessing.StandardScaler().fit(X_train_small)
#     X_train_small = scaler.transform(X_train_small)
#     X_test_small = scaler.transform(X_test_small)
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

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
#     y_train_small = vectorize_sequences(y_train_small, 3)
#     y_test_small = vectorize_sequences(y_test_small, 3)

#     X_train.append(X_train_small)
#     X_test.append(X_test_small)
#     y_train.append(y_train_small)
#     y_test_oh.append(y_test_oh_small)
#     y_test.append(y_test_small)

#########################################################################################################################################

#########################################################################################################################################

if data_type == 0:

    # for m in range(len_mean):

    #     print("\n\n* computing ", m+1, "/", len_mean, "*")
        # for i in range(len(len_layers)):
            
        #     print("** computing for", len_layers[i], " neurons **")
        #     for j in range(len(activations)):

        #         print("*** computing", activations[j],"activation ***")
        #         for l in range(len(activations)):

        #                 for k in range(len(optimizers)):
                            
                            a = time.process_time()
                            model = Sequential([
                                layers.Dense(40, activation="selu"),
                                layers.Dense(40, activation="elu"),
                                layers.Dense(40, activation="selu"),
                                layers.Dense(3, activation="softmax")
                            ])
                            model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
                            history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=10, validation_data=(X_test, y_test), verbose=0)
                            max_current_acc = max(history.history["val_accuracy"])
                            min_current_loss = min(history.history["val_loss"])
                            max_current_auc = max(history.history["val_auc"])
                            b = time.process_time()

                            if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
                                max_acc = max_current_acc
                                val_acc_function = history.history["val_accuracy"]
                                train_acc_function = history.history["accuracy"]
                                min_time_acc = b - a
                                max_parameter_acc = [40, "selu", "elu", "rmsprop"]
                            
                            if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
                                min_loss = min_current_loss
                                val_loss_function = history.history["val_loss"]
                                train_loss_function = history.history["loss"]
                                min_time_loss = b - a
                                max_parameter_loss = [40, "selu", "elu", "rmsprop"]
                            
                            if max_current_auc > max_auc or (max_current_auc == 0 and min_time_auc > b - a):
                                max_auc = max_current_auc
                                min_time_auc = b - a
                                max_parameter_auc = [40, "selu", "elu", "rmsprop"]
                                val_auc_function = history.history["val_auc"]
                                train_auc_function = history.history["auc"]
                            
                            backend.clear_session()

elif data_type == 1:

    # for m in range(len_mean):

    #     print("\n\n****** computing ", m+1, "/", len_mean, "******")
    #     for i in range(len(len_layers)):
            
    #         print("####", len_layers[i], "neurons ####")
    #         for j in range(len(activations)):

    #             print("$$", activations[j],"activation $$")
    #             for l in range(len(activations)):

    #                 print("$", activations[l], "second activation $")
    #                 for k in range(len(optimizers)):
                        
                        for p in range(len(X_train)):

                            print("seed", p)
                            a = time.process_time()
                            model = Sequential([
                                layers.Dense(40, activation="selu"),
                                layers.Dense(40, activation="elu"),
                                layers.Dense(40, activation="selu"),
                                layers.Dense(3, activation="softmax")
                            ])
                            model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
                            history = model.fit(X_train[p], y_train[p], epochs=n_epochs, batch_size=10, validation_data=(X_test[p], y_test[p]), verbose=0)
                            max_current_acc = max(history.history["val_accuracy"])
                            min_current_loss = min(history.history["val_loss"])
                            max_current_auc = max(history.history["val_auc"])
                            b = time.process_time()

                            if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
                                max_acc = max_current_acc
                                val_acc_function = history.history["val_accuracy"]
                                train_acc_function = history.history["accuracy"]
                                min_time_acc = b - a
                                max_parameter_acc = [40, "selu", "elu", "rmsprop", p]
                            
                            if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
                                min_loss = min_current_loss
                                val_loss_function = history.history["val_loss"]
                                train_loss_function = history.history["loss"]
                                min_time_loss = b - a
                                max_parameter_loss = [40, "selu", "elu", "rmsprop", p]
                            
                            if max_current_auc > max_auc or (max_current_auc == 0 and min_time_auc > b - a):
                                max_auc = max_current_auc
                                min_time_auc = b - a
                                max_parameter_auc = [40, "selu", "elu", "rmsprop", p]

                            backend.clear_session()
    
                        max_parameter_acc.append(reference[max_parameter_acc.pop()])
                        max_parameter_loss.append(reference[max_parameter_loss.pop()])
                        max_parameter_auc.append(reference[max_parameter_auc.pop()])

elif data_type == 2:

    # for m in range(len_mean):

    #     print("\n\n* computing ", m+1, "/", len_mean, "*")
    #     for i in range(len(len_layers)):
            
    #         print("** computing for", len_layers[i], " neurons **")
    #         for j in range(len(activations)):

    #             print("*** computing", activations[j],"activation ***")
    #             for l in range(len(activations)):

    #                 for k in range(len(optimizers)):
                        
                        for p in range(len(X_train)):

                            print("seed", p)
                            a = time.process_time()
                            model = Sequential([
                                layers.Dense(40, activation="selu"),
                                layers.Dense(40, activation="elu"),
                                layers.Dense(40, activation="selu"),
                                layers.Dense(3, activation="softmax")
                            ])
                            model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
                            callback = [callbacks.ModelCheckpoint("Melanoma_trained_model_seed" + str(p) + "_v4.keras", save_best_only=True)]
                            history = model.fit(X_train[p], y_train[p], epochs=60, batch_size=20, verbose=0, validation_data=(X_test[p], y_test[p]), callbacks=callback)
                            max_current_acc = max(history.history["val_accuracy"])
                            min_current_loss = min(history.history["val_loss"])
                            max_current_auc = max(history.history["val_auc"])
                            b = time.process_time()

                            if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
                                max_acc = max_current_acc
                                val_acc_function = history.history["val_accuracy"]
                                train_acc_function = history.history["accuracy"]
                                min_time_acc = b - a
                                max_parameter_acc = [40, "selu", "elu", "rmsprop", p]
                            
                            if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
                                min_loss = min_current_loss
                                val_loss_function = history.history["val_loss"]
                                train_loss_function = history.history["loss"]
                                min_time_loss = b - a
                                max_parameter_loss = [40, "selu", "elu", "rmsprop", p]

                            if max_current_auc > max_auc or (max_current_auc == 0 and min_time_auc > b - a):
                                max_auc = max_current_auc
                                min_time_auc = b - a
                                max_parameter_auc = [40, "selu", "elu", "rmsprop", p]
                                val_auc_function = history.history["val_auc"]
                                train_auc_function = history.history["auc"]

                            backend.clear_session()
    

# else:

#     for m in range(len_mean):

#         print("\n\n* computing ", m+1, "/", len_mean, "*")
#         for i in range(len(len_layers)):
            
#             print("** computing for", len_layers[i], " neurons **")
#             for j in range(len(activations)):

#                 print("*** computing", activations[j],"activation ***")
#                 for l in range(len(activations)):

#                     for k in range(len(optimizers)):
                        
#                         for p in range(len(X_train)):

#                             a = time.process_time()
#                             model = Sequential([
#                                 layers.Dense(len_layers[i], activation=activations[j]),
#                                 layers.Dense(len_layers[i], activation=activations[l]),
#                                 layers.Dense(len_layers[i], activation=activations[j]),
#                                 layers.Dense(1, activation="sigmoid")
#                             ])
#                             model.compile(optimizer=optimizers[k], loss="binary_crossentropy", metrics=["accuracy", metrics.AUC()])
#                             history = model.fit(X_train[p], y_train[p], epochs=n_epochs, batch_size=10, validation_data=(X_test[p], y_test[p]), verbose=0)
#                             max_current_acc = max(history.history["val_accuracy"])
#                             min_current_loss = min(history.history["val_loss"])
#                             max_current_auc = max(history.history["val_auc"])
#                             b = time.process_time()

#                             if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
#                                 max_acc = max_current_acc
#                                 val_acc_function = history.history["val_accuracy"]
#                                 train_acc_function = history.history["accuracy"]
#                                 min_time_acc = b - a
#                                 max_parameter_acc = [len_layers[i], activations[j], activations[l], optimizers[k], p]
                            
#                             if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
#                                 min_loss = min_current_loss
#                                 val_loss_function = history.history["val_loss"]
#                                 train_loss_function = history.history["loss"]
#                                 min_time_loss = b - a
#                                 max_parameter_loss = [len_layers[i], activations[j], activations[l], optimizers[k], p]
                            
#                             if max_current_auc > max_auc or (max_current_auc == 0 and min_time_auc > b - a):
#                                 max_auc = max_current_auc
#                                 min_time_auc = b - a
#                                 max_parameter_auc = [len_layers[i], activations[j], activations[l], optimizers[k], p]

#                             backend.clear_session()


# print("seed :", seed)
print("Best parameters for minimum loss (", min_loss, ") :", max_parameter_loss)
print("Best parameters for maximum accuracy (", max_acc, "):", max_parameter_acc)
print("Best parameter for max auc (", max_auc, "):", max_parameter_auc)

epochs = range(1, n_epochs + 1)
plt.plot(epochs, train_loss_function, "b", label="Training Loss")
plt.plot(epochs, val_loss_function, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, train_acc_function, "b", label="Training accuracy")
plt.plot(epochs, val_acc_function, "r", label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, train_auc_function, "b", label="Training auc")
plt.plot(epochs, val_auc_function, "r", label="Validation auc")
plt.title("Training and Validation auc")
plt.xlabel("Epochs")
plt.ylabel("auc")
plt.legend()
plt.show()