#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:09:23 2021

@author: krishna.kottakki
"""

import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
import mglearn 
import matplotlib.pyplot as plt 


# In[1.0]:
boston = load_boston() 
print ('data structure:', boston.data.shape)

print ('loading boaton datasets')
X, y = mglearn.datasets.load_extended_boston()
print ("X.shape", format(X.shape))

# In[2.0]:
print ('understanding KNN Classification using mglearn:')
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

# In[3.0]:
X,y = mglearn.datasets.make_forge()

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print ("training data: {}" .format(X_train.shape))
print ("training data: {}" .format(X_test.shape))

print ("training data: {}" .format(y_train.shape))
print ("training data: {}" .format(y_test.shape))

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

y_predict = knn_classifier.predict(X_test)
final_score = knn_classifier.score(X_test, y_test)
print('final score:', final_score)

# In[3.1]:
fig, axes = plt.subplots(1,3,figsize=(10,3))
for n_neighbors, ax in zip([1,3,9], axes):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(knn_classifier, X_train, fill = True, eps = 0.1, ax = ax, alpha = 0.4)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, ax=ax)
    ax.set_title("{}.neighbour(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

# In[3.1]:

from sklearn.datasets import load_breast_cancer

cancer_data_df = load_breast_cancer()
X_df = cancer_data_df.data 
y_df = cancer_data_df.target

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, stratify = y_df, random_state =10)

print ("train data - input {}".format(X_train.shape))
print ("train data - output {}".format(y_train.shape))
print ("test data - input {}".format(X_test.shape))
print ("test data - output {}".format(y_test.shape))

n_neighbours_settings = list(range(1,11))
training_accuracy = []
test_accuracy = []
count = 1
for n_neighbours in n_neighbours_settings:
    # model building 
    print("number of nighbours:", n_neighbours)
    knn_classifier = KNeighborsClassifier(n_neighbors=count).fit(X_train, y_train)
    print ([n_neighbours, knn_classifier.score(X_train,y_train)])
    training_accuracy.append(knn_classifier.score(X_train,y_train))
    test_accuracy.append(knn_classifier.score(X_test,y_test))
    count +=1 
    

plt.plot(n_neighbours_settings,training_accuracy, label="training_accuracy")
plt.plot(n_neighbours_settings,test_accuracy, label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbours")
plt.legend()

# In[4.1]
knn_classifier = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
print ([n_neighbours, knn_classifier.score(X_train,y_train)])
