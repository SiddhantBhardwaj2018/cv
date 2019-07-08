
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

#Reading the top 5 headings
df.head()

#Changed category of column "Timestamp" and assigned it to new column "Date"

df["Date"] = pd.to_datetime(df["Timestamp"],unit = "s").dt.date
group = df.groupby("Date")
data = group["Volume_(BTC)"].mean()

#Checking for null values

data.isnull().sum()
data.head()

#Setting up train and test data

train_data = data.iloc[:len(data)-30]
test_data = data.iloc[len(train_data):]

train_data = np.array(train_data)
train_data = train_data.reshape(train_data.shape[0],1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(train_data)

timestep = 30
x_train = []
y_train = []

for i in range(timestep,scaled_train.shape[0]):
    x_train.append(scaled_train[i-timestep:i,0])
    y_train.append(scaled_train[i,0])
    
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
print("x_train shape=",x_train.shape)
print("y_train shape",y_train.shape)

#Implementing Recurrent Neural Network

from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Dropout,Flatten

regressor = Sequential()
#first RNN layer
regressor.add(SimpleRNN(128,activation = "relu",return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))
#second RNN layer
regressor.add(SimpleRNN(256,activation = "relu",return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))
#third RNN layer
regressor.add(SimpleRNN(512,activation = "relu",return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.35))
#fourth RNN layer
regressor.add(SimpleRNN(256,activation = "relu",return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))
#fifth RNN layer
regressor.add(SimpleRNN(128,activation = "relu",return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))
#convert the matrix to one-line
regressor.add(Flatten())
#output layer
regressor.add(Dense(1))

regressor.compile(optimizer="adam",loss = "mse")
regressor.fit(x_train,y_train,epochs=100,batch_size=64)

inputs = data[len(data)-len(test_data)-timestep:]
inputs = inputs.values.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

predicted_data = regressor.predict(x_test)
predicted_data = scaler.inverse_transform(predicted_data)

data_test = np.array(test_data)
data_test = data_test.reshape(len(data_test),1)

#Visualizing results

plt.figure(figsize = (8,4))
plt.plot(data_test,color = "r",label = "true result")
plt.plot(predicted_data,color = "b",label = "predicted result")
plt.legend(loc = "best")
plt.xlabel("Time")
plt.ylabel("Bitcoin Volume")
plt.show()
