# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('HealthData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

#handling missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,11:13])
X[:,11:13]=imputer.transform(X[:,11:13])


#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#exploring the dataset has been done using EDA

## KNN(k-nearest neighbors algorithm) ##
#fitting the model to the training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)

#Saving the model to disk
from sklearn.externals import joblib
# filename = 'Logistic_regression_model.pkl'
filename = 'KNN.pkl'
joblib.dump(knn,filename)

#Predict the test set results
Y_pred_knn = knn.predict(X_test)


#checking the accuracy for predicted results
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred_knn)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_knn)

#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_knn))

## PREDICTION FOR NEW DATASET ##
Newdataset = pd.read_csv('newdata.csv')
ynew=knn.predict(Newdataset)
