from tensorflow.keras import Model, layers, Input, metrics
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
# 3 features


# for line in file:
#     l = line.split(",")
#     data.append(l[1:2] + l[18:20])
#     target.append(l[17])

# data = np.array(num_list2(data[1:]))
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

n_layers = range(1, 7)

filename = os.path.basename(__file__)[:-13]
path_model = Path("model")
path_perf = Path("perf")


f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_acc.txt"), "a")
f_acc.close()
f_acc = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_acc.txt"))

f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_loss.txt"), "a")
f_loss.close()
f_loss = open(path_perf / ("perf_" + filename + "_" + str(n_layers) + "_layers_" + str(split_size) + "_split_best_model_loss.txt"))

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


max_parameter_acc = -1
max_parameter_loss = -1

inputs = Input(shape=(22, 1))
x = layers.SeparableConv1D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.SeparableConv1D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy", metrics.AUC()])
history = model.fit(X_train, y_train, epochs=30, batch_size=20, verbose=0, validation_data=(X_test, y_test))

max_acc = max(history.history["val_accuracy"])
min_loss = min(history.history["val_loss"])
max_auc = max(history.history["val_auc"])

print(max_acc, max_auc)

epochs = range(1, 31)
plt.plot(epochs, history.history["loss"], "b", label="Training Loss")
plt.plot(epochs, history.history["val_loss"], "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, history.history["accuracy"], "b", label="Training accuracy")
plt.plot(epochs, history.history["val_accuracy"], "r", label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

