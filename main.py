# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:27:49 2019

@author: abrye
"""

#Dataset used can be found in this link, you need to download the data set first and  
dataset_path = "YOUR DATA SET PATH" #Example: "data/household_power_consumption.txt"
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from matplotlib import pyplot


def load(filename):
    with open(filename, 'r', encoding ='utf-8') as file:
        data = pd.read_csv(file, delimiter = ";")
    return data


data = load(dataset_path)

def preprocess(data):
  #drop rows with nan
  data = data.dropna()
  #Drop date-time
  data = data.drop(data.columns[[0,1]], axis =1)
  #prepare label by shifting the column 'Global_active_power'
  label = data['Global_active_power'][1:]
  #Delete the last row to match shape of output label
  data = data[:-1]
  return data, label
 
 
X, y = preprocess(data)
 
def train_test_splitter(X,y):
  #first 75% for training and last 25% for testing
  cutoff = int(0.75*len(X))
  X_train = X[:cutoff]
  X_test = X[cutoff:]
  y_train = y[:cutoff]
  y_test = y[cutoff:]
  return np.asarray(X_train,dtype= np.float32), np.asarray(y_train,dtype= np.float32), np.asarray(X_test,dtype= np.float32), np.asarray(y_test,dtype= np.float32)

X_train, y_train, X_test, y_test = train_test_splitter(X,y)

time_steps = 1
no_of_features = 7
# define LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(time_steps,no_of_features)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
#reshape X_train to feed into our model
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#Train the model with epochs = 5
model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, shuffle=False)

#reshape X_test to feed into our model and predict the values of Global_active_power from given X_test
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
prediction = model.predict(X_test)

pyplot.plot(y_test[:500], c = 'blue', label="actual data")
pyplot.plot(prediction[:500], c = 'red', label="prediction result")
pyplot.ylabel('Global_active_power')
pyplot.xlabel('Time step[first 500 samples of prediction taken for plotting]')
pyplot.legend()
pyplot.show()




