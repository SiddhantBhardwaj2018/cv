
#Import numpy,seaborn,pandas,matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading the csv file 
df = pd.read_csv("mushrooms.csv")

#Reading the top 5 headings
df.head()

#Using info and describe functions

df.info()
df.describe()

#Defined X and y 

X = df.iloc[:,1:]
y = df.iloc[:,0]

#Reading the shape of X and y 

X.shape
y.shape

#Applying Label Encoder on categorical variables

from sklearn.preprocessing import LabelEncoder
lb_x = LabelEncoder()
for col in X.columns:
    X[col] = lb_x.fit_transform(X[col])
lb_y = LabelEncoder()
y = lb_y.fit_transform(y)

X.head()
y

#Created dummy variables to encode all the categorical variables using one_hot_encoding approach

X = pd.get_dummies(data = X,columns = X.columns)
X.head()

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Standardizing the train and test datasets

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(X_train)
sc_x.fit(X_test)

#Applying Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Importing accuracy_score,classification_report,confusion_matrix

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#Created a function which prints the classification_report,accuracy_score and confusion_matrix from the test data and predictions data


def generate_score(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy Score:",accuracy_score(y_test,y_pred))

#Applying Logistic Regression and achieved accuracy of 86.6%

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("For Logistic Regression")
generate_score(y_test,y_pred)

#Applying RandomForestClassifier and achieved accuracy of 95.8%

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
y_pred_1  = forest.predict(X_test)
print("For Random Forest")
generate_score(y_test,y_pred_1)

#Applying DecisionTreeClassifier and achieved accuracy of 94.6%

from sklearn.tree import DecisionTreeClassifier
tree_ = DecisionTreeClassifier()
tree_.fit(X_train,y_train)
y_pred_2 = tree_.predict(X_test)
print("For DecisionTree")
generate_score(y_test,y_pred_2)

#Applying GaussianNB and achieved accuracy of 87.4%

from sklearn.naive_bayes import GaussianNB
bayes_ = GaussianNB()
bayes_.fit(X_train,y_train)
y_pred_3 = bayes_.predict(X_test)
print("For Gaussian NB")
generate_score(y_test,y_pred_3)

#Applying KNeighborsClassifier and achieved accuracy of 95.6%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 5,metric='minkowski',p=2)
knn.fit(X_train,y_train)
y_pred_4 = knn.predict(X_test)
print("For K-Nearest Neighbors")
generate_score(y_test,y_pred_4)

#Applying SVM and achieved accuracy of 95.6%

from sklearn import svm
SVM = svm.SVC()
SVM.fit(X_train,y_train)
y_pred_5 = SVM.predict(X_test)
print("For Support Vector Machine")
generate_score(y_test,y_pred_5)