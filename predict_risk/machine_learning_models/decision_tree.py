# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HealthData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 13].values


#handling missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,11:13])
X[:,11:13]=imputer.transform(X[:,11:13])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Exploration of dataset has been done using EDA 

## Decision Tree Classifier ##
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

from sklearn.externals import joblib
filename = 'decision_tree.pkl'
joblib.dump(DT,filename)

# Predicting the Test set results
y_pred_dt = DT.predict(X_test)

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_dt)

##CONFUSION MATRIX
from sklearn.metrics import classification_report, confusion_matrix  
cm=confusion_matrix(y_test, y_pred_dt) 

#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_dt))


##PREDICTION FOR NEW DATASET

Newdataset = pd.read_csv('newdata.csv')
ynew=DT.predict(Newdataset)

