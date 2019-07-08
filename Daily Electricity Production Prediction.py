
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("Electric_Production.csv")
df.rename(columns = {"DATE":"Date","IPG2211A2N":"Daily Electricity Production"},inplace = True)

#Reading the top 5 headings
df.head()

#Reading the bottom 5 headings
df.tail()

#Changed category of column "Date"

df["Date"] = pd.to_datetime(df["Date"],infer_datetime_format = True)
df = df.set_index(["Date"])

from datetime import datetime
df.head()

#Plotting Daily Production

plt.figure(figsize = (12,8))
plt.ylabel("Daily Electricity Production")
plt.xlabel("Date")
plt.plot(df)
plt.show()

#Applying the ARIMA Model

#Perform Rolling Statistic

rolmean = df.rolling(window = 30).mean()
rolstd = df.rolling(window = 30).std()
print(rolmean,rolstd)

#Plotting Original graph,Rolling Mean,Rolling Standard Deviation

plt.figure(figsize = (12,8))
orig = plt.plot(df,color = "blue",label = "original")
mean = plt.plot(rolmean,color = "red",label = "Rolling Mean")
std = plt.plot(rolstd, color = "gray",label = "Rolling STD")
plt.legend(loc = "best")
plt.title("Rolling Mean & Standard Deviation")
plt.show()

#Perform Dickey- Fuller Test
from statsmodels.tsa.stattools import adfuller

print("Results of Dickey-Fuller Test")
dftest = adfuller(df["Daily Electricity Production"],autolag = "AIC")

dfoutput = pd.Series(dftest[0:4],index = ["Test-Statistic","p-value","#Lags Used","Number of Observations used"])
for key,value in dftest[4].items():
    dfoutput["Critical Value (%s)"%key] = value
    
print(dfoutput)

#Plotting seasonality,trend and residuals

plt.figure(figsize = (20,10))
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df,label = "Original")
plt.legend(loc = "best")
plt.subplot(412)
plt.plot(trend,label = "Trend")
plt.legend(loc = "best")
plt.subplot(413)
plt.plot(seasonal ,label = "seasonal")
plt.legend(loc = "best")
plt.subplot(414)
plt.plot(residual,label = "Residuals")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()

#Plotting Autocorrelation function and Partial Autocorrelation function

plt.figure(figsize = (20,10))
from statsmodels.tsa.stattools import acf,pacf

lag_acf = acf(df,nlags = 20)
lag_pacf = pacf(df,nlags = 20,method = "ols")

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0,linestyle = "--",color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.title("AutoCorrelation Function")

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0,linestyle = '--',color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.title("Partial AutoCorrelation Function")

#Since the data has a seasonal component, I shall be using SARIMAX for time series analysis

from statsmodels.tsa.statespace.sarimax import SARIMAX

train_data = df[:len(df)-30]
test_data = df[len(df)-30:]

arima_model = SARIMAX(train_data["Daily Electricity Production"],order = (3,2,3),seasonal_order = (2,1,2,30))
arima_result = arima_model.fit()
arima_result.summary()

#Obtaining Predictions

arima_pred = arima_result.predict(start = len(train_data),end = len(df)-1,typ =  "levels").rename("ARIMA Predictions")
arima_pred

#Plotting the test_data against arima_pred

test_data["Daily Electricity Production"].plot(figsize=(16,5),legend = True)
arima_pred.plot(legend = True)

#Checking for accuracy
'''
importing rmse
'''

from statsmodels.tools.eval_measures import rmse

'''
Obtaining mse error = 17.45380820970808, rmse = 17.45380820970808, and mean =  88.84721763224181
'''

arima_rmse_error = rmse(test_data["Daily Electricity Production"],arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df["Daily Electricity Production"].mean()

print(f"MSE Error: {arima_mse_error}\nRMSE Error:{arima_mse_error}\nMean: {mean_value}")

# Applying LSTM Model

#importing MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

#Implementing model

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 30
n_features = 1
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data,length=n_input,batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM,Dense

lstm_model = Sequential()
lstm_model.add(LSTM(200,activation = "relu",input_shape = (n_input,n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer = "adam",loss = "mse")

lstm_model.summary()

lstm_model.fit_generator(generator,epochs = 20)

losses_lstm = lstm_model.history.history["loss"]
plt.figure(figsize = (12,4))
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);

lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape(1,n_input,n_features)

for i in range(len(test_data)):
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis = 1)
    
lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions_scaled

test_data["LSTM_Predictions"] = lstm_predictions

test_data

#Visualizing results

test_data["Daily Electricity Production"].plot(figsize = (16,5),legend = True)
test_data["LSTM_Predictions"].plot(legend = True)

'''
Obtained rmse = 6.128012336278951, mse = 37.55253519358701, mean = 88.84721763224181
'''

lstm_rmse_error = rmse(test_data["Daily Electricity Production"],test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df["Daily Electricity Production"].mean()
print(f"MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}")



