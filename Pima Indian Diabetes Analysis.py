
#Import numpy,seaborn,pandas,matplotlib
import numpy as pp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("diabetes.csv")

#Reading the top 5 headings
df.head()

df.describe()
df.info()

#Exploratory Data Analysis
'''
Drawing a barplot and a piechart highlighting describing proportion of population with diabetes
'''

f, ax = plt.subplots(1, 2, figsize = (15, 7))
f.suptitle("Diabetes?", fontsize = 18.)
_ = df.Outcome.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])
_ = df.Outcome.value_counts().plot.pie(labels = ("No", "Yes"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\
colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")

'''
Drawing a violinplot comparing presence of diabetes with number of pregnancies
'''

sns.violinplot(x = "Outcome",y = "Pregnancies",data = df)

'''
Drawing a violinplot comparing presence of diabetes with amount of glucose
'''

sns.violinplot(x = "Outcome",y = "Glucose",data = df)

'''
Drawing a violinplot comparing presence of diabetes with bloodpressure
'''

sns.violinplot(x = "Outcome",y = "BloodPressure",data = df)

'''
Drawing a violinplot comparing presence of diabetes with skin thickness
'''

sns.violinplot(x = "Outcome",y = "SkinThickness",data = df)

'''
Drawing a violinplot comparing presence of diabetes with level of insulin
'''

sns.violinplot(x = "Outcome",y = "Insulin",data = df)

'''
Drawing a violinplot comparing presence of diabetes with BMI
'''

sns.violinplot(x = "Outcome",y = "BMI",data = df)

'''
Drawing a violinplot comparing presence of diabetes with Diabetes Pedigree Function
'''

sns.violinplot(x = "Outcome",y = "DiabetesPedigreeFunction",data = df)

'''
Drawing a violinplot comparing presence of diabetes with Age
'''

sns.violinplot(x = "Outcome",y = "Age",data = df)

#Made a distplot of Age column of the dataframe

sns.distplot(df.Age,bins = 20)

#Made a distplot of Diabetes Pedigree Function column of the dataframe

sns.distplot(df.DiabetesPedigreeFunction,bins = 20)

#Made a distplot of BMI column of the dataframe

sns.distplot(df.BMI,bins = 20)

#Made a distplot of Insulin column of the dataframe

sns.distplot(df.Insulin,bins = 20)

#Made a distplot of Skin Thickness column of the dataframe

sns.distplot(df.SkinThickness,bins = 20)

#Made a distplot of Blood Pressure column of the dataframe

sns.distplot(df.BloodPressure,bins = 20)

#Made a distplot of Glucose column of the dataframe

sns.distplot(df.Glucose,bins = 20)

#Made a distplot of Pregnancies column of the dataframe

sns.distplot(df.Pregnancies,bins = 20)

#Created a scatter matrix of the dataset

from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize = (15,20))
plt.show()

# Using Machine Learning Techniques for classification of persons into categories of whether or not they have diabetes

'''
Importing various classifier methods
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

'''
Importing various scores to measure accuracy of classification methods
'''

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

'''
Created a function which prints the classification_report,accuracy_score and confusion_matrix from the test data and predictions
data
'''

def generate_score(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy Score:",accuracy_score(y_test,y_pred))
    
'''
Defined X and y
'''

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

'''
Importing train_test_split function
'''

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=0)

'''
Applied LogisticRegression technique and got accuracy_score of 82%
'''

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("For Logistic Regression")
generate_score(y_test,y_pred)

'''
Applied GaussianNB classifier technique and got accuracy_score of 77.8%
'''

bayes_ = GaussianNB()
bayes_.fit(X_train,y_train)
y_pred_1 = bayes_.predict(X_test)
print("For Gaussian NB")
generate_score(y_test,y_pred_1)

'''
Applied KNeighborsClassifier technique and got accuracy_score of 75.3%
'''

knn = KNeighborsClassifier(n_neighbors= 5,metric='minkowski',p=2)
knn.fit(X_train,y_train)
y_pred_2 = knn.predict(X_test)
print("For K-Nearest Neighbors")
generate_score(y_test,y_pred_2)

'''
Applied Support Vector Machine technique and achieved accuracy_score of 67.9%
'''

SVM = svm.SVC()
SVM.fit(X_train,y_train)
y_pred_3 = SVM.predict(X_test)
print("For Support Vector Machines")
generate_score(y_test,y_pred_3)

'''
Applied DecisionTreeClassifier technique and achieved accuracy_score of 75%
'''

tree_ = DecisionTreeClassifier()
tree_.fit(X_train,y_train)
y_pred_4 = tree_.predict(X_test)
print("For Decision Trees")
generate_score(y_test,y_pred_4)

'''
Applied RandomForestClassifier technique and achieved accuracy_score of 76.5%
'''

forest = RandomForestClassifier()
forest.fit(X_train,y_train)
y_pred_5 = forest.predict(X_test)
print("For Random Forests")
generate_score(y_test,y_pred_5)