#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:58:13 2018

@author: dongdongmary
"""

#import all package
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#import iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
print( X.shape, y.shape)

# Split the dataset into a training and a testing set
# Test set will be the 10% taken randomly
random_state = np.arange(1,11,1).tolist()
in_sample_accuracy = []
out_of_sample_accuracy = []

for n in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.1, 
                                                        random_state = n)
    ## Standardize the features
    #sc = StandardScaler()
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_test_std = sc.transform(X_test)
    
    #Decision tree model
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    
    #score
    in_sample_accuracy.append(accuracy_score(y_train,y_pred_train))
    out_of_sample_accuracy.append(accuracy_score(y_test,y_pred_test))
    print()
    print(in_sample_accuracy)
    print(out_of_sample_accuracy)
 

#dataframe
tree_scores = pd.DataFrame({'random_state':random_state,
                       'in_sample_accuracy':in_sample_accuracy,
                       'out_of_sample_accuracy':out_of_sample_accuracy})
tree_scores_stat = tree_scores.describe()
tree_scores_stat = tree_scores_stat.drop(['count', 'min','25%', '50%','75%', 'max'])
tree_scores_stat.loc['mean','random_state'] = ''
tree_scores_stat.loc['std','random_state'] = ''
print(tree_scores)
print()



#cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.1, 
                                                    random_state = 1)
cv_socres = cross_val_score(tree, X_train, y_train, cv=10)
print('Cross Validation scores:',cv_socres.tolist())
print()
print('Mean of CV scores:',cv_socres.mean())
print()
print('Std of CV scores:',cv_socres.std())
print()

y_pred = tree.predict(X_test)
scores = accuracy_score(y_test, y_pred) 
print('out_sample_accuracy: ',scores)

#
print('--------------------------------------------------------------------')
print("My name is Yuezhi Li")
print("My NetID is: yuezhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
