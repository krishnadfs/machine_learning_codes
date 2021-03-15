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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz # to import visualizations ffor decision trees 

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 

import os

# In[Data sources:]
cancer_data = load_breast_cancer()
boston = load_boston() 
print ('data structure:', boston.data.shape)
print ('loading boaton datasets')
X, y = mglearn.datasets.load_extended_boston()
print ("X.shape", format(X.shape))

# In[Functions:]
def plot_feature_importances_cancer(model):   
    n_features = cancer_data.data.shape[1]    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer_data.feature_names,size=8)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    

# In[2.0]:
print ('understanding KNN Classification using mglearn:')
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

# In[3.0]:
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print ("training data: {}" .format(X_train.shape))
print ("training data: {}" .format(X_test.shape))

print ("training data: {}" .format(y_train.shape))
print ("training data: {}" .format(y_test.shape))


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

# In[4.1] KNN regression on make_wave data 

X_df,y_df  = mglearn.datasets.make_wave()
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state =0)

knn_reg = KNeighborsRegressor(n_neighbors = 3)
knn_reg_model = knn_reg.fit(X_train, y_train)


# In[4.2]: model analysis 

fig, axes = plt.subplots(1,3,figsize=(18,3))
line = np.linspace(-3, 3, 1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9], axes):
    print (n_neighbors)
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
    ax.plot(line, knn_regressor.predict(line))
    ax.plot(X_train, y_train, '^', c= mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c= mglearn.cm2(0), markersize=8)
    ax.set_title("{} neighbo(s)\n train_score: {:.2f} test_score: {:.2f}".format(
        n_neighbors, knn_regressor.score(X_test,y_test), 
        knn_regressor.score(X_train,y_train)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    del knn_regressor
    

axes[0].legend(["Model predictions", "training data/target", "test data/target"], 
               loc ="best")


# In[4.3]: Linear Regression with OLS 



# X_make_wave, y_make_wave = mglearn.datasets.make_wave()
X_exd_boston, y_exd_boston = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X_exd_boston, y_exd_boston, random_state =0)

lin_reg = LinearRegression().fit(X_train,y_train)
print ("lin_reg.Coeff_ :{}".format(lin_reg.coef_))
print ("lin_reg.Intercept_ :{}".format(lin_reg.intercept_))

print("training set score: {: 2f}".format(lin_reg.score(X_train, y_train)))
print("test set score: {: 2f}".format(lin_reg.score(X_test, y_test)))


lin_reg_ridge_1 = Ridge(alpha =1 ).fit(X_train,y_train)
print ("lin_reg.Coeff_ :{}".format(lin_reg_ridge_1.coef_))
print ("lin_reg.Intercept_ :{}".format(lin_reg_ridge_1.intercept_))

print("training set score: {: 2f}".format(lin_reg_ridge_1.score(X_train, y_train)))
print("test set score: {: 2f}".format(lin_reg_ridge_1.score(X_test, y_test)))


lin_reg_ridge_2 = Ridge(alpha =0.1).fit(X_train,y_train)
print ("lin_reg.Coeff_ :{}".format(lin_reg_ridge_2.coef_))
print ("lin_reg.Intercept_ :{}".format(lin_reg_ridge_2.intercept_))

print("training set score: {: 2f}".format(lin_reg_ridge_2.score(X_train, y_train)))
print("test set score: {: 2f}".format(lin_reg_ridge_2.score(X_test, y_test)))

# In[4.3]: Linear Regression with OLS 


lin_reg_lasso_1 = Lasso(alpha =0.0001 ).fit(X_train,y_train)
print ("lin_reg.Coeff_ :{}".format(lin_reg_lasso_1.coef_))
print ("lin_reg.Intercept_ :{}".format(lin_reg_lasso_1.intercept_))

print("training set score: {: 2f}".format(lin_reg_lasso_1.score(X_train, y_train)))
print("test set score: {: 2f}".format(lin_reg_lasso_1.score(X_test, y_test)))

# In[4.3]: Linear Classification using support vector machines 


c_parameter = 100
cancer_data_df = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data_df.data, 
                                                    cancer_data_df.target,
                                                    stratify = cancer_data_df.target, 
                                                    random_state =0)

log_reg_model = LogisticRegression(C=c_parameter).fit(X_train, y_train)
print("training set score: {: 2f}".format(log_reg_model.score(X_train, y_train)))
print("test set score: {: 2f}".format(log_reg_model.score(X_test, y_test)))

# In[4.0]: Linear Classification using support vector machines 

make_blobs_df = make_blobs(random_state=42)

X_data, y_data = make_blobs_df
mglearn.discrete_scatter(X_data[:,0], X_data[:,1], y_data)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])

linear_svm = SVC(kernel="linear").fit(X_data, y_data)

print("coefficient shape:", linear_svm.coef0)
print("intercept shape:", linear_svm.intercept_)

 
mglearn.discrete_scatter(X_data[:,0], X_data[:,1], y_data)
line = np.linspace(-15,15)

for (coeff_, intercept, color) in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line*coeff_[0]+intercept)/coeff_[1])
    # print([coeff, intercept, color])

# In[5.0]: decision tree - Classifier / Regression 

cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer_data.data, 
                                                                                cancer_data.target,
                                                                                stratify = cancer_data.target, 
                                                                                random_state = 42)
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_clf.fit(cancer_X_train, cancer_y_train)

print("training set score: {: 2f}".format(tree_clf.score(cancer_X_train, cancer_y_train)))
print("test set score: {: 2f}".format(tree_clf.score(cancer_X_test, cancer_y_test)))

export_graphviz(tree_clf, out_file = "tree_clf.dot", 
                class_names = cancer_data.target_names, 
                feature_names=cancer_data.feature_names, 
                impurity=False, filled = True)
# with open("tree_clf.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))

# In[5.0]: decision tree - Classifier / Regression 

print ("Feature importances:\n{}".format(tree_clf.feature_importances_))
plot_feature_importances_cancer(tree_clf)

tree = mglearn.plots.plot_tree_not_monotone()
# pd.display(tree)

# In[5.0]: decision tree - Regression on historical RAM prices 


ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
ram_prices.describe()

plt.semilogy(ram_prices.date, ram_prices.price, 'r-')
plt.xlabel("Year")
plt.ylabel("Price is $/Mbyte")


# In[]:

# splitting the data into train and test data 
data_train = ram_prices[ram_prices.date <= 2000]
data_test = ram_prices[ram_prices.date > 2000]

X_train = data_train.date[:,np.newaxis]
y_train = np.log(data_train.price)

tree_reg = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree_reg.predict(X_all)
pred_lin = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lin)

plt.semilogy(data_train.date, data_train.price, label = "Training data")
plt.semilogy(data_test.date, data_test.price, label = "Test data")
plt.semilogy(ram_prices.date, price_tree, label = "Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label = "Lin_reg prediction")
plt.legend()

# In[]:

X, y = make_moons(noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)

random_forest_with_5_trees = RandomForestClassifier(n_estimators=5, random_state=2)
random_forest_with_5_trees.fit(X_train,y_train)

fig, axes = plt.subplots(2,3,figsize=(20,10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), random_forest_with_5_trees.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(random_forest_with_5_trees, X_train, fill =True,
                                ax=axes[1,-1], alpha =0.4)
axes[1,-1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)


# In[]:
random_forest_100_trees = RandomForestClassifier(n_estimators=100, random_state=2)
random_forest_100_trees.fit(cancer_X_train, cancer_y_train)

print("Accuracy on training set {}".format(random_forest_100_trees.score(cancer_X_train, 
                                                                         cancer_y_train)))

print("Accuracy on test set {}".format(random_forest_100_trees.score(cancer_X_test, 
                                                                         cancer_y_test)))

plot_feature_importances_cancer(random_forest_100_trees)

# In[]:

handcraft_X_data, handcraft_y_data = mglearn.tools.make_handcrafted_dataset()
model_svm = SVC(kernel = "rbf", C= 10, gamma =0.1).fit(handcraft_X_data, handcraft_y_data)

mglearn.plots.plot_2d_separator(model_svm, handcraft_X_data, eps=0.5)
mglearn.discrete_scatter(handcraft_X_data[:,0], handcraft_X_data[:,1],handcraft_y_data)

sv = model_svm.support_vectors_
sv_labels = model_svm.dual_coef_.ravel() >0 

mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=10, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


fig, axes = plt.subplots(3,3,figsize =(15,10))

for ax, C in zip(axes, [-1,0,3]):
    for a , gamma in zip(ax, range(-1,2)):
        mglearn.plots.plt_svm(log_c=C, log_gamma=gamma, ax=a)
        
axes[0,0] legend(["class 0", "class 1", "sv classs 0", "sv class 1", ncol=4, 
                  loc = (0.9, 1,2))])      
