from tensorflow.keras import Sequential, layers, Model, Input
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



max_acc = 0
min_time_acc = inf
min_loss = inf
min_time_loss = inf


max_parameter_acc = (-1, -1, -1)
max_parameter_loss = (-1, -1, -1)

##############################################################################################################################################

##############################################################################################################################################
# fitting and evaluating


len_mean = 30
n_epochs = 80
# len_layers = [40, 50, 60, 80, 100, 150]
# activations = ["relu", "selu", "elu"]
# optimizers = ["rmsprop", "SGD"]
# # batch_size = [5, 10, 15, 20]

len_layers = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
activations = ["relu", "selu", "elu", "exponential"]
optimizers = ["rmsprop", "SGD", "adam", "adadelta", "Adagrad", "adamax", "nadam", "Ftrl"]
batch_size = [5, 10, 15, 20, 30, 40, 50, 80, 100, 150]


for m in range(len_mean):

    print("* computing", m+1, "/", len_mean, "*")
    a = time.process_time()
    inputs = Input(shape=(22,1))
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(filters=32, kernel_size=5, use_bias=False, activation="relu")
    for size in [32, 64, 128]:
        residual = x

        # x = layers.BatchNormalization()(x)
        x = layers.Conv1D(size, 3, use_bias=False, padding="same")(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(size, 3, use_bias=False, padding="same", activation="relu")(x)

        x = layers.MaxPooling1D(3, strides=2, padding="same")(x)

        residual = layers.Conv1D(size, 1, strides=2, padding="same", use_bias=False)(residual)
        x = layers.add([x, residual])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=15, validation_data=(X_test, y_test), verbose=0)
    max_current_acc = max(history.history["val_accuracy"])
    min_current_loss = min(history.history["val_loss"])
    b = time.process_time()

    if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
        max_acc = max_current_acc
        val_acc_function = history.history["val_accuracy"]
        train_acc_function = history.history["accuracy"]
        min_time_acc = b - a
    
    if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
        min_loss = min_current_loss
        val_loss_function = history.history["val_loss"]
        train_loss_function = history.history["loss"]
        min_time_loss = b - a


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

