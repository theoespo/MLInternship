from tensorflow.keras import Sequential, layers, metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from math import inf
import time, os
from pathlib import Path

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
            data_num[i].append(float(data[i][j]))
    return data_num

file = open("parkinsons.data")
data = []
target = []

##############################################################################################################################################

#############################################################################################################################################
# All data

for line in file:
    l = line.split(",")
    data.append(l[1:17] + l[18:])
    target.append(l[17])

data = np.array(num_list2(data[1:]))
target = np.array(num_list(target[1:]))

##############################################################################################################################################
# All data + PCA


# for line in file:
#     l = line.split(",")
#     data.append(l[1:2] + l[4:5] + l[9:10] + l[15:16] + l[18:21])
#     target.append(l[17])

# data = PCA().fit_transform(num_list2(data[1:]))
# target = np.array(num_list(target[1:]))

##############################################################################################################################################

##############################################################################################################################################
# single split of the data

split_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=split_size, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##############################################################################################################################################

##############################################################################################################################################
# initiate quantities

diff = 1
n_layers = 3

filename = os.path.basename(__file__)[:-13]
path_model = Path("model")
path_perf = Path("perf")

if diff == 0:
    f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_acc.txt"), "a")
    f_acc.close()
    f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_acc.txt"))

    f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_loss.txt"), "a")
    f_loss.close()
    f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_loss.txt"))
else:
    f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_diff_" + str(split_size) + "_split_best_model_acc.txt"), "a")
    f_acc.close()
    f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_diff_" + str(split_size) + "_split_best_model_acc.txt"))

    f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_diff_" + str(split_size) + "_split_best_model_loss.txt"), "a")
    f_loss.close()
    f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_diff_" + str(split_size) + "_split_best_model_loss.txt"))

prev_perf = []
for line in f_acc:
    l = line.split(",")
    for i in range(len(l)):
        prev_perf += [float(l[i])]
f_acc.close()

for line in f_loss: 
    l = line.split(",")
    for i in range(len(l)):
        prev_perf += [float(l[i])]

if len(prev_perf) != 0:
    max_acc = prev_perf[0]
    min_time_acc = prev_perf[1]
    min_loss = prev_perf[2]
    min_time_loss = prev_perf[3]
else:
    max_acc = 0
    min_time_acc = inf
    min_loss = inf
    min_time_loss = inf


max_parameter_acc = (-1, -1, -1)
max_parameter_loss = (-1, -1, -1)

##############################################################################################################################################

##############################################################################################################################################
# fitting and evaluating


len_mean = 1
n_epochs = 15
len_layers = [40, 50, 60, 80, 100, 150]
activations = ["relu", "selu", "elu"]
optimizers = ["rmsprop", "SGD"]
# batch_size = [5, 10, 15, 20]

# len_layers = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
# activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
# optimizers = ["rmsprop", "SGD", "adam", "adadelta", "Adagrad", "adamax", "nadam", "Ftrl"]
# batch_size = [5, 10, 15, 20, 30, 40, 50, 80, 100, 150]


for m in range(len_mean):

    print("\n\n* computing ", m+1, "/", len_mean, "*")
    for i in range(len(len_layers)):
        
        print("** computing for", len_layers[i], " neurons **")
        for j in range(len(activations)):

            print("*** computing", activations[j],"activation ***")
            for l in range(len(activations)):

                for k in range(len(optimizers)):
                    
                    a = time.process_time()
                    model = Sequential([
                        layers.Dense(len_layers[i], activation=activations[j]),
                        layers.Dense(len_layers[i], activation=activations[l]),
                        layers.Dense(len_layers[i], activation=activations[j]),
                        layers.Dense(1, activation="sigmoid")
                    ])
                    model.compile(optimizer=optimizers[k], loss="binary_crossentropy", metrics=["accuracy", metrics.AUC()])
                    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=10, validation_data=(X_test, y_test), verbose=0)
                    max_current_acc = max(history.history["val_accuracy"])
                    min_current_loss = min(history.history["val_loss"])
                    b = time.process_time()

                    if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
                        max_acc = max_current_acc
                        val_acc_function = history.history["val_accuracy"]
                        train_acc_function = history.history["accuracy"]
                        min_time_acc = b - a
                        if diff == 0:
                            max_parameter_acc = len_layers[i], activations[j], optimizers[k]
                            model.save(path_model / (filename + "_" + str(len(model.layers) - 1) + "_layers_" + str(split_size) + "_split_Best_Model_acc"))
                            f = open(path_perf / ("perf_" + filename + "_" + str(len(model.layers) - 1) + "_layers_" + str(split_size) + "_split_best_model_acc.txt"), "w")
                            f.write(str(max_acc) + ", " + str(min_time_acc))
                            f.close()
                        else:
                            max_parameter_acc = len_layers[i], activations[j], activations[l], activations[j], optimizers[k]
                            model.save(path_model / (filename + "_" + str(len(model.layers) - 1) + "_layers_diff_" + str(split_size) + "_split_Best_Model_acc"))
                            f = open(path_perf / ("perf_" + filename + "_" + str(len(model.layers) - 1) + "_layers_diff_" + str(split_size) + "_split_best_model_acc.txt"), "w")
                            f.write(str(max_acc) + ", " + str(min_time_acc))
                            f.close()
                    
                    if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
                        min_loss = min_current_loss
                        val_loss_function = history.history["val_loss"]
                        train_loss_function = history.history["loss"]
                        min_time_loss = b - a
                        if diff == 0:
                            max_parameter_loss = len_layers[i], activations[j], optimizers[k]
                            model.save(path_model / (filename + "_" + str(len(model.layers) - 1) + "_layers_" + str(split_size) + "_split_Best_Model_loss"))
                            f = open(path_perf / ("perf_" + filename + "_" + str(len(model.layers) - 1) + "_layers_" + str(split_size) + "_split_best_model_loss.txt"), "w")
                            f.write(str(min_loss) + ", " + str(min_time_loss))
                            f.close()
                        else:
                            max_parameter_loss = len_layers[i], activations[j], activations[l], activations[j], optimizers[k]
                            model.save(path_model / (filename + "_" + str(len(model.layers) - 1) + "_layers_diff_" + str(split_size) + "_split_Best_Model_loss"))
                            f = open(path_perf / ("perf_" + filename + "_" + str(len(model.layers) - 1) + "_layers_diff_" + str(split_size) + "_split_best_model_loss.txt"), "w")
                            f.write(str(min_loss) + ", " + str(min_time_loss))
                            f.close()


##############################################################################################################################################
# Best models

print("Best parameters for minimum loss (", min_loss, ") :", max_parameter_loss)
print("Best parameters for maximum accuracy (", max_acc, "):", max_parameter_acc)

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

