# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:07:40 2022

@author: HP
"""

#Import necessary packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os

#1. Read CSV data
file_path = r"C:\Users\HP\Documents\TensorFlow\datasets\heart.csv"
heart_data = pd.read_csv(file_path)

#3. Split the data into features and label
heart_features = heart_data.copy()
heart_label = heart_features.pop('target')

#4. Check the data
print("------------------Features-------------------------")
print(heart_features.head())
print("-----------------Label----------------------")
print(heart_label.head())

#%%
#5. One hot encode label
heart_label_OH = pd.get_dummies(heart_label)
#Check the one-hot label
print("--------------------One-hot Label-----------------")
print(heart_label_OH.head())

#6. Split the features and labels into train-validation-test sets
#Using 60:20:20 split
SEED = 12345
x_train, x_iter, y_train, y_iter = train_test_split(heart_features,heart_label_OH,test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#7. Normalize the features, fit with training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#Data preparation is completed at this step

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = x_train.shape[-1]
number_output = y_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input)) 
model.add(tf.keras.layers.Dense(64,activation='elu'))
model.add(tf.keras.layers.Dense(32,activation='elu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(number_output,activation="softmax"))

#9.Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#%%
#10. Train and evaluate model
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\HP\Documents\Program AI06\p1_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE=32
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb_callback,es_callback])

#%%
#Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")