import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import main_model
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('disease_dataset.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
#
# cm = confusion_matrix(y_test, y_pred)


def predict_disease(filename):
    result = main_model.disease(filename)
    result = [result[1:4]]
    result_pd = pd.DataFrame(result)
    print result_pd
    result_pd = sc.transform(result_pd)
    print result_pd
    y_pred = classifier.predict(result_pd)
    result = y_pred[0]
    if(result == 0):
        return 'Common Rust'
    else:
        return 'Southern Leaf Blight'
