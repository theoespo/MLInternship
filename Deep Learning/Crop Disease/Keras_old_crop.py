import random
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt

def path_to_input_image(path):
    return img_to_array(load_img(path))

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(int(data[i]))
    return data_num

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        results[i, sequences[i]] = 1.
    return results


details = open("kaggle/input/Crop_details.csv", "r")
data_path = []
target = []
for line in details:
    l = line.split(",")
    data_path.append(l[1][1:])
    target.append(l[-1][:-1])

data_path = np.array(data_path[1:799])
target = np.array(num_list(target[1:799]))

random.Random(1337).shuffle(data_path)
random.Random(1337).shuffle(target)

####################################################################################################################################

n_epochs = 20
img_size = (224, 224)
num_img_input = len(data_path)

target_vector = vectorize_sequences(target, 5)
# target_vector = np.zeros(shape=(num_img_input, 5))
# for i in range(num_img_input):
#     target_vector[i, target[i]] = 1

data = np.zeros((num_img_input,) + img_size + (3,), dtype="float32")
for i in range(num_img_input):
    data[i] = path_to_input_image(data_path[i])

num_val_sample = int(0.25*num_img_input)
train_data = data[:-num_val_sample].reshape((len(data[:-num_val_sample]), 224*224*3))
train_target = target_vector[:-num_val_sample]
test_data = data[-num_val_sample:].reshape((len(data[-num_val_sample:]), 224*224*3))
test_target = target_vector[-num_val_sample:]

####################################################################################################################################

model = Sequential([
                        layers.Dense(16, activation="relu"),
                        layers.Dense(32, activation="relu"),
                        layers.Dense(64, activation="relu"),
                        layers.Dense(5, activation="softmax")
                    ])
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()
history = model.fit(train_data, train_target, epochs=n_epochs, batch_size=10, validation_data=(test_data, test_target))
max_current_acc = max(history.history["val_accuracy"])
min_current_loss = min(history.history["val_loss"])

val_acc_function = history.history["val_accuracy"]
train_acc_function = history.history["accuracy"]

val_loss_function = history.history["val_loss"]
train_loss_function = history.history["loss"]

####################################################################################################################################

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

