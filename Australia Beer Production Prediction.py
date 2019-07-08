
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 

df = pd.read_csv("monthly-beer-production-in-austr.csv")

#Reading the top 5 headings

df.head()

#Changed category of column "Month"

df["Month"] = pd.to_datetime(df["Month"],infer_datetime_format=True)
df = df.set_index(["Month"])
from datetime import datetime
df.head()

#Plotting Monthly Production

plt.figure(figsize = (12,8))
plt.xlabel("Date")
plt.ylabel("Amount of Beer Production")
plt.plot(df)
plt.show()

#Applying the ARIMA Model

#Perform Rolling Statistic
rolmean = df.rolling(window=12).mean()
rolstd = df.rolling(window=12).std()
print(rolmean,rolstd)

#Plotting Original graph,Rolling Mean,Rolling Standard Deviation

orig = plt.plot(df,color = "blue",label = "original")
mean = plt.plot(rolmean,color = "red",label = "Rolling Mean")
std = plt.plot(rolstd,color = "gray",label = "Rolling STD")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation")
plt.show(block=False)

#Perform Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

print("Results of Dickey-Fuller Test")
dftest = adfuller(df["Monthly beer production"],autolag = "AIC")

dfoutput = pd.Series(dftest[0:4],index = ["Test-Statistic","p-value","#Lags Used","Number of Observations used"])
for key,value in dftest[4].items():
    dfoutput["Critical Value (%s)"% key] = value
    
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
plt.plot(seasonal,label = "seasonality")
plt.legend(loc = "best")
plt.subplot(414)
plt.plot(residual,label = "Residuals")
plt.legend(loc="best")
plt.tight_layout()

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
plt.title("Autocorrelation Function")

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0,linestyle = "--",color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(df)),linestyle = "--",color = "gray")
plt.title("Partial Autocorrelation Function")

#Since the data has a seasonal component, I shall be using SARIMAX for time series analysis

from statsmodels.tsa.statespace.sarimax import SARIMAX
train_data = df[:len(df)-12]
test_data = df[len(df)-12:]
arima_model = SARIMAX(train_data["Monthly beer production"],order = (2,1,1),seasonal_order = (4,0,3,12))
arima_result = arima_model.fit()
arima_result.summary()

#Obtaining Predictions

arima_pred = arima_result.predict(start = len(train_data),end = len(df)-1,typ = "levels").rename("ARIMA Predictions")
arima_pred

#Plotting the test_data against arima_pred

test_data["Monthly beer production"].plot(figsize = (16,5),legend = True)
arima_pred.plot(legend = True)

#Checking for accuracy
'''
importing rmse
'''

from statsmodels.tools.eval_measures import rmse

'''
Obtaining mse error = 66.11050409066426, rmse = 66.11050409066426, and mean =  136.39537815126045
'''

arima_rmse_error = rmse(test_data["Monthly beer production"],arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df["Monthly beer production"].mean()

print(f"MSE Error: {arima_mse_error}\nRMSE Error: {arima_mse_error}\nMean: {mean_value}")

# Applying LSTM Model

#importing MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

#Implementing model

from keras.models import Sequential
from keras.layers import LSTM,Dense

lstm_model = Sequential()
lstm_model.add(LSTM(200,activation = "relu",input_shape = (n_input,n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer = "adam",loss = "mse")

lstm_model.summary()

lstm_model.fit_generator(generator,epochs = 20)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
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

test_data["LSTM_Predictions"] = lstm_predictions

test_data

#Visualizing results

test_data["Monthly beer production"].plot(figsize = (16,5),legend = True)
test_data["LSTM_Predictions"].plot(legend = True)

'''
Obtained rmse = 10.587254501359485, mse = 112.08995787655668, mean = 136.39537815126045
'''

lstm_rmse_error = rmse(test_data["Monthly beer production"],test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df["Monthly beer production"].mean()

print(f"MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}")


