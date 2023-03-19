# importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.activations import swish
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras

# Read the file to get the values for the labels
data = pd.read_csv('test/filteredmachine (7) (3) (2).csv')
data = pd.DataFrame(data)
label = data['Attack'].values.tolist()

vocab = tf.range(len(set(label)), dtype=tf.int64)

# Create a lookup table
table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list(set(label)), vocab), -1)

# Use the lookup table to map labels to integers
int_labels = table.lookup(tf.constant(label))

# One-hot encode the integer labels
one_hot_labels = tf.one_hot(int_labels, depth=4)

#### need to understand why need to remove this
# title = data.drop(index=0)

# print(data)

# the remaining data left are removing is the sample
# data2 = data["Logs"].values
# data2 = pd.DataFrame(data2)

#### need to convert the dataset to numbers using this link
#### https://www.youtube.com/watch?v=ORpDAUQUnkU
#### attempt to train the data with no error
## if this work can clear line 20 and 21
data3 = data["Logs"].values
vectorizer = TfidfVectorizer()
vectorizer.fit(data3)
tfidf_matrix = vectorizer.transform(data3)

# test1 = tf.strings.to_number(data3, out_type=tf.int64)
# my_int_tensor = tf.cast(test1, dtype=tf.int)

print("---------------------1----------------------")

# print(label.shape)
# print(data3.shape)
# print(my_int_tensor)


# Converts the lists to a tensor object
#label = tf.strings.to_number(label, tf.float64)
train_labels = tf.convert_to_tensor(one_hot_labels)

sparse_tensor = tf.sparse.SparseTensor(indices=np.mat([tfidf_matrix.nonzero()[0], tfidf_matrix.nonzero()[1]]).transpose(),
                                       values=tfidf_matrix.data.astype(np.float64),
                                       dense_shape=tfidf_matrix.shape)

#train_samples = tf.sparse.to_dense(sparse_tensor)
train_samples = tf.sparse.reorder(sparse_tensor)

print(train_samples.shape)
# normalization_layer = tf.keras.layers.Normalization(axis=1)
# train_samples = normalization_layer(train_samples)

print("---------------------2----------------------")

# print(train_labels)
# print(train_samples)

print("---------------------3----------------------")

# Create the model using sequential the identify the layers it should go through
# model = keras.Sequential([
#     # 1st layer mostly to specify the input shape
#     # Dense(units=64, input_shape=(None, 1143), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dense(units=64, input_shape=(None, 555), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#     Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#     Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     # Dropout(0.5),
#     # 3rd layer is to split the data into either 1 of the 5 units
#     Dense(units=5, activation='softmax')
# ])
#
model = keras.Sequential([
    # 1st layer mostly to specify the input shape
    # Dense(units=32, input_shape=(None, 1143), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(units=4, input_shape=(None, 592)),
    Dense(units=12, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(units=16),
    Dense(units=16),
    Dense(units=16),
    Dense(units=16),
    Dense(units=12),
    # Dense(units=16, activation='relu'),
    # 3rd layer is to split the data into either 1 of the 4 units
    Dense(units=4, activation='softmax')
])

# regularization[l2] 0.01, epochs 100 = 170
# regularization[l2] 0.01, epochs 100 (2.50am)= 73
# regularization[l2] 0.1, epochs 150 = 3
# regularization[l2] 0.05, epochs 150 = 4
# regularization[l1] 0.01, epochs 100 = 3

# none, regularization[l1] 0.01, regularization[l2] 0.01, epochs 145 = 74
# none, regularization[l1] 0.01, regularization[l2] 0.01, epochs 150 = 104
# none, regularization[l1] 0.01, regularization[l2] 0.01, epochs 155 = 79
# none, regularization[l1] 0.01, regularization[l2] 0.01, epochs 160 = 79

# regularization[l2] 0.001, regularization[l2] 0.001, regularization[l2] 0.001, epochs 100 = 85
# regularization[l2] 0.001, regularization[l2] 0.001, epochs 100 = 100
# regularization[l2] 0.001, regularization[l2] 0.001, epochs 100 = 100


#### need to try running with a loss and without to see which has better accuracy
# This will compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# This will train the model with the given datasets
model.fit(train_samples, train_labels, batch_size=592, epochs=150, verbose=1, shuffle=True)

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(tfidf_matrix, label, epochs=10, batch_size=1)


# This will save the model that was used to train
# model 6 is ideal
# model 7 is good
# model 8 is test
model.save("model/training_17.h5")

