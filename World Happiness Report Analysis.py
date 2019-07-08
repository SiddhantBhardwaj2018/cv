
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("2015.csv")

#Reading the top 5 headings
df.head()

#Reading the bottom 5 headings
df.tail()

#Checking for any null values
df.isnull().any()

#Using info function

df.info()

#Created an lmplot comparing Happiness Score with Economy (GDP per Capita)

sns.lmplot(x = "Happiness Score",y = "Economy (GDP per Capita)",data = df)

#Created an lmplot comparing Happiness Score with Family

sns.lmplot(x = "Happiness Score",y = "Family",data = df)

#Created an lmplot comparing Happiness Score with Health (Life Expectancy)

sns.lmplot(x = "Happiness Score",y = 'Health (Life Expectancy)',data = df)

#Created an lmplot comparing Happiness Score with Freedom

sns.lmplot(x = "Happiness Score",y = 'Freedom',data = df)

#Created an lmplot comparing Happiness Score with Trust (Government Corruption)

sns.lmplot(x = "Happiness Score",y = 'Trust (Government Corruption)',data = df)

#Created an lmplot comparing Happiness Score with Generosity

sns.lmplot(x = "Happiness Score",y = "Generosity",data = df)

#Created an lmplot comparing Happiness Score with Dystopia Residual

sns.lmplot(x = "Happiness Score",y = "Dystopia Residual",data = df)

#Creating correlation heatmap

x_df = df.drop(columns = ["Happiness Rank","Standard Error"],axis = 1)
x_df.corr()

plt.figure(figsize = (12,8))
sns.heatmap(x_df.corr(),cmap =  "inferno",annot = True)
plt.show()

#Finding mean across various categories across regions

group = x_df.groupby(by = "Region")
new_df = group.mean()
new_df

#Finding median across various categories across regions

new1_df = group.median()
new1_df

#Created barplot comparing region with happiness scores

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Happiness Score"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Happiness Score",fontsize = 20)

#Created barplot comparing region with Economy (GDP per Capita)

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Economy (GDP per Capita)"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Economy (GDP per Capita)",fontsize = 20)

#Created barplot comparing region with Family

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Family"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Family",fontsize = 20)

#Created barplot comparing region with Health (Life Expectancy)

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Health (Life Expectancy)"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Health (Life Expectancy)",fontsize = 20)

#Created barplot comparing region with Freedom

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Freedom"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Freedom",fontsize = 20)

#Created barplot comparing region with Trust (Government Corruption)

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Trust (Government Corruption)"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Trust (Government Corruption)",fontsize = 20)

#Created barplot comparing region with Generosity

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Generosity"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Generosity",fontsize = 20)

#Created barplot comparing region with Dystopia Residual

plt.figure(figsize = (30,12))
sns.barplot(x = x_df["Region"],y = x_df["Dystopia Residual"],data = df)
plt.xlabel("Region",fontsize = 20)
plt.ylabel("Dystopia Residual",fontsize = 20)

#Created scatterplot comparing Economy (GDP per Capita) with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Economy (GDP per Capita)",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Family with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Family",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Health (Life Expectancy) with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Health (Life Expectancy)",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Freedom with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Freedom",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Trust (Government Corruption) with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Trust (Government Corruption)",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Generosity with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Generosity",y = "Happiness Score",data = df,hue = "Region")

#Created scatterplot comparing Dystopia Residual with Happiness Score across Regions

plt.figure(figsize = (12,8))
sns.scatterplot(x = "Dystopia Residual",y = "Happiness Score",data = df,hue = "Region")

#Defining X and y 

X = df.drop(["Country","Region","Happiness Rank","Happiness Score"],axis = 1)
y = df["Happiness Score"]

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Importing mean_absolute_error,mean_squared_error,r2_score

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#Applied Linear Regression technique and achieved accuracy of 99%

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("accuracy is: "+ str(lr.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred)))
print("R Squared: {}".format(r2_score(y_test,y_pred)))


#Applied Ridge Regression technique and achieved accuracy of 99%

rig = linear_model.Ridge()
rig.fit(X_train,y_train)
y_pred_1 = rig.predict(X_test)
print("Accuracy is: "+ str(rig.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_1)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_1)))
print("R Squared: {}".format(r2_score(y_test,y_pred_1)))

#Applied DecisionTreeRegressor technique and achieved accuracy of 78%

from sklearn.tree import DecisionTreeRegressor
tree_ = DecisionTreeRegressor()
tree_.fit(X_train,y_train)
y_pred_2 = tree_.predict(X_test)
print("Accuracy is: "+ str(tree_.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_2)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_2)))
print("R Squared: {}".format(r2_score(y_test,y_pred_2)))

#Applied AdaBoostRegressor technique and achieved accuracy of 84%

from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(loss = "exponential")
ada.fit(X_train,y_train)
y_pred_3 = ada.predict(X_test)
print("Accuracy is: "+ str(ada.score(X_test,y_test) * 100) + "%")
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test,y_pred_3)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_pred_3)))
print("R Squared: {}".format(r2_score(y_test,y_pred_3)))
















