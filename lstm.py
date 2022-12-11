import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""#Read Dataset"""

data= pd.read_csv('IBM.csv')

data.head()

"""#Data Preprocessing
The close value of the stock is chosen as the data to train our model. Data is scaled using MinMaxScaler. The dataset consist of 755 datapoints which is divided to train and test datasets. First 600 is selected as train dataset and other 155 datapoinst as test dataset.
"""

data['Date']= pd.to_datetime(data.Date)

feature = data.iloc[:, 4:5]

scale= MinMaxScaler(feature_range=(0, 1))

feature = scale.fit_transform(feature)

data.info()

train = feature[:600]
test = feature[600:]

"""The previous 30 days data will be used as an input for the model to predict the value of next day. That's why the training data is divided to the two parts using this script. Furthermore, for the model we need to have 3D data and because of that it is reshaped."""

train_x=[]
train_y=[]

for i in range(30, 600):
  train_x.append(train[i-30:i, 0])
  train_y.append(train[i, 0])

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x.shape

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))


"""
The test dataset is also preprocessed as training dataset for testing the performance of the model
"""

test_x=[]
test_y=[]

for i in range(30, 155):
  test_x.append(test[i-30:i, 0])
  test_y.append(test[i, 0])

test_x = np.array(test_x)
test_y = np.array(test_y)

test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

if __name__ == '__main__':
  """#Model Development"""
  model = tf.keras.models.Sequential()
  
  model.add(tf.keras.models.LSTM (units=100, return_sequences= True, input_shape = (train_x.shape[1], 1)))
  model.add(tf.keras.models.Dropout(rate=0.2))
  
  model.add(tf.keras.models.LSTM (units=100, return_sequences= True))
  model.add(tf.keras.models.Dropout(rate=0.2))
  
  model.add(tf.keras.models.LSTM (units=100, return_sequences= True))
  model.add(tf.keras.models.Dropout(rate=0.2))
  
  model.add(tf.keras.models.LSTM (units=100, return_sequences= True))
  model.add(tf.keras.models.Dropout(rate=0.2))
  
  model.add(tf.keras.models.LSTM (units=100))
  model.add(tf.keras.models.Dropout(rate=0.2))
  
  model.add(tf.keras.models.Dense(units=1))
  
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.summary()
  
  """#Model Training """
  
  history= model.fit(train_x, train_y, epochs=100, batch_size=16)
  loss = history.history['loss']
  
  plt.plot(loss)

  """ Prediction """
  predict_y = model.predict(test_x)

  predict_y =np.reshape(predict_y, (predict_y.shape[0], predict_y.shape[1]))
  test_y =np.reshape(test_y, (test_y.shape[0], 1))

  """The result is transformed to initial form of dataset for getting appropriate value."""

  predict_y = scale.inverse_transform(predict_y)
  test_y = scale.inverse_transform(test_y)

  plt.plot(predict_y, color='red')
  plt.plot(test_y, color='blue')
  plt.show()