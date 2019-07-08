
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#Reading the csv file 
df =pd.read_csv("insurance.csv")

#Reading the top 5 headings
df.head()

#Changing columns to category
df["region"] = df["region"].astype("category")
df["smoker"] = df["smoker"].astype("category")
df["sex"] = df["sex"].astype("category")

#Using info and describe functions

df.info()
df.describe()

#Created a boxplot comparing region with charges

plt.figure(figsize = (10,6))
sns.boxplot(x = "region",y = "charges",data = df)
plt.show()

#Created a boxplot comparing smoker with charges

plt.figure(figsize = (10,6))
sns.boxplot(x = "smoker",y = "charges",data = df)
plt.show()

#Created a boxplot comparing sex with charges

plt.figure(figsize = (10,6))
sns.boxplot(x = "sex",y = "charges",data = df)
plt.show()

#Created a barplot comparing no. of children with charges

plt.figure(figsize = (12,6))
sns.barplot(x = "children",y = "charges",data = df)

#Created a distplot for bmi column 

plt.figure(figsize = (12,6))
sns.distplot(df["bmi"],bins = 40,kde = True)
plt.show()

#Created a distplot for bmi column 

plt.figure(figsize = (12,6))
sns.distplot(df["charges"],bins = 100,kde = True)
plt.show()

#Created a jointplot comparing bmi with charges

sns.jointplot(x = "bmi",y = "charges",data = df,kind = "hex",color = "magenta")

#Created a jointplot comparing age with charges
 
sns.jointplot(x = "age",y = "charges",data = df,kind = "hex",color = "steelblue")

#Created dummy variables to encode all the categorical variables using one_hot_encoding approach

one_hot_encoders_df = pd.get_dummies(df)
one_hot_encoders_df.head()

#Defined X and y 

X = one_hot_encoders_df.drop(["charges"],axis = 1)
y = one_hot_encoders_df["charges"]

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =  0.33,random_state = 42)

#Standardizing the train and test datasets

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(X_train)
sc_x.transform(X_test)

#Importing multiple Regression ML Techniques

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

#Importing mean_absolute_error,mean_squared_error,r2_score

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

#Applied Linear Regression technique and achieved accuracy of 76%

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_train)
print("Accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")

#Applied Polynomial Regression technique and achieved accuracy of 85.54%

quad = PolynomialFeatures(degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(x_quad,y,random_state = 0)
plr = LinearRegression().fit(X_train,y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print("Accuracy is: "+ str(plr.score(X_test,y_test) * 100) + "%")

#Applied RandomForestRegressor technique and achieved accuracy of 87.40%

forest = RandomForestRegressor(n_estimators=100,criterion = "mse",random_state=1,n_jobs = -1)
forest.fit(X_train,y_train)
y_pred_1 = forest.predict(X_test)
print("Accuracy is: "+ str(forest.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_1)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_1)))
print("R Squared: {}".format(r2_score(y_test,y_pred_1)))

#Applied AdaBoostRegressor technique and achieved accuracy of 87.53%

ada = AdaBoostRegressor()
ada.fit(X_train,y_train)
y_pred_3 = ada.predict(X_test)
print("Accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_3)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_3)))
print("R Squared: {}".format(r2_score(y_test,y_pred_3)))