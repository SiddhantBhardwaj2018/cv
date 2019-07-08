
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("kc_house_data.csv")

#Reading the top 5 headings
df.head()

#Dropping "id" and "date" columns
df = df.drop(["id","date"],axis = 1)

#Checking for any null values

df.isnull().sum()

#Using info and describe functions

df.info()
df.describe()

#Creating heatmap of correlation between different features

df.corr()
plt.figure(figsize = (20,10))
sns.heatmap(df.corr(),annot =True)
plt.show()

#Created a jointplot comparing sqft_living with price 

sns.jointplot(x = "sqft_living",y = "price",data = df,kind = "reg")

#Created a jointplot comparing sqft_lot with price 

sns.jointplot(x = "sqft_lot",y = "price",data = df,kind = "reg")

#Created a jointplot comparing no.of bedrooms with price 

sns.jointplot(x = "bedrooms",y = "price",data = df,kind = "reg")

#Created a jointplot comparing no.of bathrooms with price 

sns.jointplot(x = "bathrooms",y = "price",data = df,kind = "reg")

#Created a boxplot comparing view with price

plt.figure(figsize = (12,6))
sns.boxplot(x = "view",y = "price",data = df)
plt.show()

#Created a boxplot comparing presence of waterfront with price

plt.figure(figsize = (10,6))
sns.boxplot(x = "waterfront",y = "price",data = df)
plt.show()

#Created a boxplot comparing no. of floors with price

plt.figure(figsize = (12,6))
sns.boxplot(x = "floors",y = "price",data = df)
plt.show()

#Created a boxplot comparing condition with price

plt.figure(figsize =(12,6))
sns.boxplot(x = "condition",y = "price",data = df)
plt.show()

#Created a jointplot comparing sqft_above with price

plt.figure(figsize =(12,6))
sns.jointplot(x = "sqft_above",y = "price",data = df,kind = "reg")
plt.show()

#Created a jointplot comparing sqft_basement with price

plt.figure(figsize =(12,6))
sns.jointplot(x = "sqft_basement",y = "price",data = df,kind = "reg")
plt.show()

#Created a jointplot comparing sqft_living15 with price

plt.figure(figsize =(12,6))
sns.jointplot(x = "sqft_living15",y = "price",data = df,kind = "reg")
plt.show()

#Created a jointplot comparing sqft_lot15 with price

plt.figure(figsize =(12,6))
sns.jointplot(x = "sqft_lot15",y = "price",data = df,kind = "reg")
plt.show()

#Created dummy variables to encode all the categorical variables using one_hot_encoding approach

one_hot_encoders_df = pd.get_dummies(df)
one_hot_encoders_df.head()

#Defined X and y 

X = one_hot_encoders_df.drop(["price"],axis = 1).values
y = one_hot_encoders_df["price"].values

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

#Standardizing the train and test datasets

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(X_train)
sc_x.transform(X_test)

#Importing mean_absolute_error,mean_squared_error,r2_score

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#Applied Linear Regression technique and achieved accuracy of 69.94%

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))
print("R Squared: {}".format(r2_score(y_test,y_pred)))

#Applied Lasso Regression technique and achieved accuracy of 69.94%

from sklearn import linear_model
las = linear_model.Lasso()
las.fit(X_train,y_train)
y_pred = las.predict(X_test)
print("accuracy is: "+ str(las.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))
print("R Squared: {}".format(r2_score(y_test,y_pred)))

#Applied Ridge Regression technique and achieved accuracy of 69.93%

rig = linear_model.Ridge()
rig.fit(X_train,y_train)
y_pred_1 = rig.predict(X_test)
print("Accuracy is: "+ str(rig.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_1)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_1)))
print("R Squared: {}".format(r2_score(y_test,y_pred_1)))

#Applied DecisionTreeRegressor technique and achieved accuracy of 72.76%

from sklearn.tree import DecisionTreeRegressor
tree_ = DecisionTreeRegressor()
tree_.fit(X_train,y_train)
y_pred_2 = tree_.predict(X_test)
print("Accuracy is: "+ str(tree_.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_2)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_2)))
print("R Squared: {}".format(r2_score(y_test,y_pred_2)))

#Applied AdaBoostRegressor technique and achieved accuracy of 40.82%

from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(loss = "exponential")
ada.fit(X_train,y_train)
y_pred_3 = ada.predict(X_test)
print("Accuracy is: "+ str(ada.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_3)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_3)))
print("R Squared: {}".format(r2_score(y_test,y_pred_3)))

#Applied XGBRegressor technique and achieved accuracy of 87.05%

from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
y_pred_8 = xgb.predict(X_test)
print("Accuracy is: "+ str(xgb.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_8)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_8)))
print("R Squared: {}".format(r2_score(y_test,y_pred_8)))