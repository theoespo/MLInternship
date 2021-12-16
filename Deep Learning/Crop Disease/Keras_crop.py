import random, os, cv2
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def path_to_input_image(data_path):
    data = np.zeros((len(data_path),) + img_size + (3,), dtype="float32")
    for i in range(len(data_path)):
        data[i] = img_to_array(load_img(data_path[i]))
    return data

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def num_list(data):
    data_num = []
    for i in range(len(data)):
        data_num.append(int(data[i]))
    return data_num

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        results[i, sequences[i]] = 1.
    return np.array(results)


details = open("kaggle/input/Crop_details.csv", "r")
data_path = []
target = []
for line in details:
    l = line.split(",")
    data_path.append(l[1][1:])
    target.append(l[-1][:-1])

data_path = np.array(data_path[1:799])
target = np.array(num_list(target[1:799]))

seed = random.randint(0, 2000)
random.Random(seed).shuffle(data_path)
random.Random(seed).shuffle(target)

####################################################################################################################################

n_epochs = 50
img_size = (224, 224)
num_img_input = len(data_path)

target_vector = vectorize_sequences(target, 5)

data = path_to_input_image(data_path)

num_val_sample = int(0.25*num_img_input)
train_data = data[:-num_val_sample]
train_target = target_vector[:-num_val_sample]
test_data = data[-num_val_sample:]
test_target = target_vector[-num_val_sample:]


####################################################################################################################################

inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False, activation="relu")(x)

for size in [32, 64, 128, 256, 512]:

    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(filters=size, kernel_size=3, padding="same", use_bias=False, activation="relu")(x)

    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(filters=size, kernel_size=3, padding="same", use_bias=False, activation="relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(filters=size, kernel_size=1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint("crop_class_best2.keras", save_best_only=True)]

history = model.fit(train_data, train_target, epochs=n_epochs, batch_size=16, validation_data=(test_data, test_target), callbacks=callbacks)

###################################################################################################################################################

max_acc = max(history.history["val_accuracy"])
min_loss = min(history.history["val_loss"])

val_acc_function = history.history["val_accuracy"]
train_acc_function = history.history["accuracy"]

val_loss_function = history.history["val_loss"]
train_loss_function = history.history["loss"]

###################################################################################################################################################

print("Minimum loss :", min_loss*100)
print("Maximum accuracy :", max_acc*100)

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


###################################################################################################################################################

model = keras.models.load_model("crop_class_best2.keras")

###################################################################################################################################################

test_dir = "kaggle/input/test_crop_image"
species = ["jute", "maize", "rice", "sugarcane", "wheat"]
new_test_data_path = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
new_test_target_fname = [fname for fname in os.listdir(test_dir)]
new_test_target = []
for f in new_test_target_fname:
    for specie in species:
        if f.startswith(specie):
            new_test_target.append(species.index(specie))
            break

new_test_data = np.zeros((len(new_test_data_path),) + img_size + (3,), dtype="float32")
for i in range(len(new_test_data_path)):
    im = img_to_array(load_img(new_test_data_path[i]))
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)
    new_test_data[i] = im

new_test_target_vector = vectorize_sequences(new_test_target, 5)

print("Test accuracy :", model.evaluate(new_test_data, new_test_target_vector)[1])
