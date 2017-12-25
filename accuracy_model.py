#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:01:07 2017

@author: satyarth
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:27:30 2017

@author: satyarth
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('disease_dataset.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, [4,5]].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#y_pred_train = classifier.predict(X_train)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_train, y_pred_train)

