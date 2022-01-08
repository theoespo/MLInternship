from tensorflow.keras import Model, layers, Input, metrics, backend, callbacks
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from math import inf
import time, os, random
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

max_acc = 0
min_time_acc = inf
min_loss = inf
min_time_loss = inf
max_auc = 0
min_time_auc = inf

max_parameter_acc = -1
max_parameter_loss = -1
max_parameter_auc = -1

seed = random.randint(0, 2000)
seed = 44
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
# All data with different random state (absurd value) Max acc : 0.8534482717514038 epochs : 48 best seed : 44, Min loss : 0.4196934401988983 epochs : 84 best seed : 44, Max auc : 0.9579184651374817 epochs : 98 best seed : 44

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

if data_type == 0:

    inputs = Input(shape=(6, 1))
    residual = inputs
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", use_bias="False", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    # x = layers.SeparableConv1D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPooling1D(pool_size=2)(x)*
    residual = layers.Conv1D(filters=32, kernel_size=1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
    callback = [callbacks.ModelCheckpoint("Melanoma_trained_model_seed44_v3.keras", save_best_only=True)]
    history = model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=0, validation_data=(X_test, y_test), callbacks=callback)

    val_acc_function = history.history["val_accuracy"]
    train_acc_function = history.history["accuracy"]

    val_loss_function = history.history["val_loss"]
    train_loss_function = history.history["loss"]

    val_auc_function = history.history["val_auc"]
    train_auc_function = history.history["auc"]

    max_acc = max(history.history["val_accuracy"])
    min_loss = min(history.history["val_loss"])
    max_auc = max(history.history["val_auc"])

if data_type == 2:

    for m in range(len(X_train)):

        print(m)
        a = time.process_time()
        inputs = Input(shape=(6, 1))
        residual = inputs
        x = layers.BatchNormalization()(inputs)
        x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", use_bias="False", padding="same")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        # x = layers.SeparableConv1D(filters=64, kernel_size=3, activation="relu")(x)
        # x = layers.MaxPooling1D(pool_size=2)(x)*
        residual = layers.Conv1D(filters=32, kernel_size=1, strides=2, padding="same", use_bias=False)(residual)
        x = layers.add([x, residual])
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(3, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", metrics.AUC()])
        callback = [callbacks.ModelCheckpoint("Melanoma_trained_model_seed" + str(m) +".keras", save_best_only=True)]
        history = model.fit(X_train[m], y_train[m], epochs=100, batch_size=20, verbose=0, validation_data=(X_test[m], y_test[m]), callbacks=callback)
        b = time.process_time()

        max_current_acc = max(history.history["val_accuracy"])
        min_current_loss = min(history.history["val_loss"])
        max_current_auc = max(history.history["val_auc"])

        if max_acc < max_current_acc or (max_acc == max_current_acc and b - a < min_time_acc):
            max_acc = max_current_acc
            val_acc_function = history.history["val_accuracy"]
            train_acc_function = history.history["accuracy"]
            min_time_acc = b - a
            max_parameter_acc = m

        if min_current_loss < min_loss or (min_current_loss == min_loss and b - a < min_time_loss):
            min_loss = min_current_loss
            val_loss_function = history.history["val_loss"]
            train_loss_function = history.history["loss"]
            min_time_loss = b - a
            max_parameter_loss = m

        if max_current_auc > max_auc or (max_current_auc == 0 and min_time_auc > b - a):
            max_auc = max_current_auc
            min_time_auc = b - a
            max_parameter_auc = m
            val_auc_function = history.history["val_auc"]
            train_auc_function = history.history["auc"]

        backend.clear_session()

#########################################################################################################################################

print("Max acc :", max_acc, "epochs :", val_acc_function.index(max_acc), "best seed :", max_parameter_acc)
print("Min loss :", min_loss, "epochs :", val_loss_function.index(min_loss), "best seed :", max_parameter_loss)
print("Max auc :", max_auc, "epochs :", val_auc_function.index(max_auc), "best seed :", max_parameter_auc)

epochs = range(1, 101)
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
plt.ylabel("Auc")
plt.legend()
plt.show()
