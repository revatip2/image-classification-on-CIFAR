import tensorflow as tf
from functools import partial
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

cifar_dataset = tf.keras.datasets.cifar10.load_data()
(X_train_full, y_train_full), (X_test, y_test) = cifar_dataset
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train.shape

X_train.dtype

y_train.shape

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=32, input_shape=[32, 32, 3]),
    tf.keras.layers.MaxPool2D((2,2)),
   # DefaultConv2D(filters=32),
    #tf.keras.layers.MaxPool2D((2,2)),
    DefaultConv2D(filters=64),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=4, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=32, input_shape=[32, 32, 3]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=64),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.summary()

model.layers

hidden1 = model.layers[1]

hidden1.name

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

'''As the question does not mention if we have to show per epoch CPU and Wall times, I have added two code blocks:
A: Returns only Final CPU and Wall times.
B: Returns per epoch CPU and Wall times.
Please run based on which is required. Keeping default A.'''

#A
start_time = time.process_time()
start_wall_time = time.time()
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
end_time = time.process_time()
end_wall_time = time.time()
cpu_time = end_time - start_time
wall_time = end_wall_time - start_wall_time
print("CPU time: {:.2f}s".format(cpu_time))
print("Wall time: {:.2f}s".format(wall_time))

#B
'''
history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
start_time_total = time.process_time()
start_wall_time_total = time.time()

for epoch in range(30):
    start_time = time.process_time()
    start_wall_time = time.time()

    epo_history = model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid))

    end_time = time.process_time()
    end_wall_time = time.time()

    cpu_time = end_time - start_time
    wall_time = end_wall_time - start_wall_time

    print("Epoch {} - CPU time: {:.2f}s".format(epoch+1, cpu_time))
    print("Epoch {} - Wall time: {:.2f}s".format(epoch+1, wall_time))

    history["loss"].extend(epo_history.history["loss"])
    history["accuracy"].extend(epo_history.history["accuracy"])
    history["val_loss"].extend(epo_history.history["val_loss"])
    history["val_accuracy"].extend(epo_history.history["val_accuracy"])


end_time_total = time.process_time()
end_wall_time_total = time.time()

cpu_time_total = end_time_total - start_time_total
wall_time_total = end_wall_time_total - start_wall_time_total

print("CPU time total: {:.2f}s".format(cpu_time_total))
print("Wall time total: {:.2f}s".format(wall_time_total))
'''

'''Based on which you run in the previous cell, I have added two code blocks:
A: Returns only Final CPU and Wall times.
B: If you ran the code which returns per epoch CPU and Wall times.
Please run based on which is required. Keeping default A.'''

#A

pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",style=["r--", "r--.", "b-", "b-*"])
plt.show()

#B
'''
pd.DataFrame(history).plot(figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",style=["r--", "r--.", "b-", "b-*"])
plt.show()
'''

model.evaluate(X_test, y_test)

y_proba = model.predict(X_test)

y_pred = np.argmax(y_proba, axis=1)

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)

precision_class = precision_score(y_test, y_pred, average=None)
recall_class = recall_score(y_test, y_pred, average=None)
for i in range(len(precision_class)):
    print(f"Class {i} {class_names[i]}:\nPrecision={precision_class[i]}\nRecall={recall_class[i]}")
