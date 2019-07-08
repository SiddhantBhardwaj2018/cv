
#The MNIST original dataset is unavailable on mldata.org The following code, referenced from https://github.com/scikit-learn/scikit-learn/issues/8588#issuecomment-292634781, helps in solving this problem.

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)

fetch_mnist()
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
mnist

# MNIST Data 

mnist.data 
mnist.target
mnist.data.shape
mnist.target.shape

#Import numpy,seaborn,pandas,matplotlib

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Plotting mnist data

plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(mnist.data[69880:69890],mnist.target[0:10])):
    plt.subplot(1,10,index + 1)
    plt.imshow(np.reshape(image,(28,28)),cmap = 'Greys')

#Importing train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(mnist.data,mnist.target,test_size = 0.3, random_state = 0 )

#Checking shape of train and test datasets 

X_train.shape
X_test.shape
y_train.shape
y_test.shape

#Applying Logistic Regression and achieved accuracy score of 92%

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial',penalty='l1', solver='saga', tol=0.1)
lr.fit(X_train, y_train)
lr.predict(X_test[432].reshape(1,-1))
predictions = lr.predict(X_test)
score = lr.score(X_test,y_test)
print(score)

#Performance Evaluation

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,predictions)
cf
plt.figure(figsize=(9,9))
sns.heatmap(cf,annot = True,cmap = 'Blues_r',linewidth = 0.5,square = True)

index = 0
misclassified = []
for wrong,right in zip(predictions,y_test):
    if wrong != right:
        misclassified.append(wrong)
        index += 1
  
plt.figure(figsize = (20,4))
for index,wrong in enumerate(misclassified[10:20]):
    plt.subplot(1,10,index + 1)
    plt.imshow(np.reshape(image,(28,28)),cmap = 'viridis')
    
#Applying Decision Tree Classifier and achieved accuracy score of 92%

from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train,y_train)
clf1.predict(X_test[432].reshape(1,-1))
predictions1 = clf1.predict(X_test)
cf1 = confusion_matrix(y_test,predictions)
cf1
score1 = lr.score(X_test,y_test)
print(score1)
plt.figure(figsize=(9,9))
sns.heatmap(cf,annot = True,cmap = 'Blues_r',linewidth = 0.5,square = True)

index = 0
misclassified = []
for wrong,right in zip(predictions,y_test):
    if wrong != right:
        misclassified.append(wrong)
        index += 1
        
plt.figure(figsize = (20,4))
for index,wrong in enumerate(misclassified[10:20]):
    plt.subplot(1,10,index + 1)
    plt.imshow(np.reshape(image,(28,28)),cmap = 'viridis')