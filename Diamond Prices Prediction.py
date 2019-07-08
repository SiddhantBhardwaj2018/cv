
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("diamonds.csv")

#Reading the top 5 headings
df.head()

#Dropping "Unnamed: 0" column 
df = df.drop(["Unnamed: 0"],axis=1)

#Using info and describe functions
df.info()
df.describe()

#Created a new dataframe where height, width , length variables are equal to 0

zero_df = df[(df["x"] == 0)|(df["y"] == 0)|(df["z"] == 0)]
zero_df

#Obtained shape of the dataframe

zero_df.shape

#Dropping the zero_df dataframe from original df dataframe

df.drop(zero_df.index,inplace = True)

#Checking for any null values

df.isnull().sum()
df.head()

#Creating heatmap of correlation between different features

df.corr()
plt.figure(figsize = (15,10))
sns.heatmap(df.corr(),annot = True)
plt.show()

#Created a boxplot comparing diamond cut with its price

plt.figure(figsize = (7,4))
sns.boxplot(x="cut",y = "price",data = df)
plt.show()

#Created a boxplot comparing diamond color with its price

plt.figure(figsize = (7,4))
sns.boxplot(x="color",y = "price",data = df)
plt.show()

#Created a boxplot comparing diamond clarity with its price

plt.figure(figsize = (7,4))
sns.boxplot(x="clarity",y = "price",data = df)
plt.show()

#Created a pairplot comparing various features of the dataframe with each other

plt.figure(figsize = (12,10))
sns.pairplot(data = df)
plt.show()

#Created a distplot of the diamond carat

colors = sns.color_palette("deep")
sns.distplot(df["carat"],color = colors[0])

#Created a distplot of the diamond depth

sns.distplot(df["depth"],color = colors[1])

#Created a distplot of the diamond table

sns.distplot(df["table"],color = colors[2] )

#Created a distplot of the diamond price

sns.distplot(df["price"],color = colors[3])

#Created a distplot of the diamond length

sns.distplot(df["x"],color = colors[4])

#Created a distplot of the diamond width

sns.distplot(df["y"],color = colors[5])

#Created a distplot of the diamond height

sns.distplot(df["z"],color = colors[6])

#Created dummy variables to encode all the categorical variables using one_hot_encoding approach

one_hot_encoders_df = pd.get_dummies(df)
one_hot_encoders_df.head()

#Defined X and y 

X = one_hot_encoders_df.drop(["price"],axis=1)
y = one_hot_encoders_df.price

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=70)

#Standardizing the train and test datasets

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(X_train)
sc_x.transform(X_test)

#Importing mean_absolute_error,mean_squared_error,r2_score

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#Applied Linear Regression technique and achieved accuracy of 91.94%

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))
print("R Squared: {}".format(r2_score(y_test,y_pred)))

#Applied Lasso Regression technique and achieved accuracy of 91.94%

from sklearn import linear_model
las = linear_model.Lasso()
las.fit(X_train,y_train)
y_pred = las.predict(X_test)
print("accuracy is: "+ str(las.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))
print("R Squared: {}".format(r2_score(y_test,y_pred)))

#Applied Ridge Regression technique and achieved accuracy of 91.95%

rig = linear_model.Ridge()
rig.fit(X_train,y_train)
y_pred_1 = rig.predict(X_test)
print("Accuracy is: "+ str(rig.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_1)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_1)))
print("R Squared: {}".format(r2_score(y_test,y_pred_1)))

#Applied DecisionTreeRegressor technique and achieved accuracy of 96.47%

from sklearn.tree import DecisionTreeRegressor
tree_ = DecisionTreeRegressor()
tree_.fit(X_train,y_train)
y_pred_2 = tree_.predict(X_test)
print("Accuracy is: "+ str(tree_.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_2)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_2)))
print("R Squared: {}".format(r2_score(y_test,y_pred_2)))

#Applied AdaBoostRegressor technique and achieved accuracy of 85.84%

from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(loss = "exponential")
ada.fit(X_train,y_train)
y_pred_3 = ada.predict(X_test)
print("Accuracy is: "+ str(ada.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_3)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_3)))
print("R Squared: {}".format(r2_score(y_test,y_pred_3)))








