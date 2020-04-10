#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:27:10 2019

@author: nolancoble
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB 
from sklearn import model_selection
from utilities import visualize_classifier

input_file = 'iris.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
print(X[:2,:])
print(y[:2])

classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)

classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train)))
print("#"*40 + "\n")
print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred))
print("#"*40 + "\n")
