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
X_train, X_test,Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

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
DT.fit(X_train,Y_train)
y_pred_dt = DT.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred_dt)

from sklearn.metrics import classification_report, confusion_matrix  
cm=confusion_matrix(Y_test, y_pred_dt) 

## KNN(k-nearest neighbors algorithm) ##
#fitting the model to the training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn = knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred_knn)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_knn)

## Naive Bayes
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,Y_train)
y_pred_nb = NB.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred_nb)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred_nb)

## RandomForest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier().fit(X_train,Y_train)
Y_pred_rf = RF.predict(X_test)
Y_pred_rf = RF.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred_rf)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rf)

#Ensemble learning

from sklearn.ensemble import VotingClassifier
estimators=[('nb', NB), ('lr', knn), ('dt', DT),('rf', RF)]

majority_voting = VotingClassifier(estimators, voting='hard')

majority_voting.fit(X_train,Y_train)
majority_voting.score(X_test, Y_test)

#Bagging classifier
from sklearn.ensemble import BaggingClassifier
Bagging= BaggingClassifier(max_samples=0.5, max_features=1.0, n_estimators=50)
Bagging.fit(X_train,Y_train)
Bagging.score(X_test, Y_test)

#Boosting
#1. Adaboost
from sklearn.ensemble import AdaBoostClassifier
AdaBoost= AdaBoostClassifier(n_estimators=500)
AdaBoost.fit(X_train,Y_train)
AdaBoost.score(X_test, Y_test)

#2 Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
GB= GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1)
GB.fit(X_train,Y_train)
GB.score(X_test, Y_test)

print('Majority Voting accuracy score: ',majority_voting.score(X_test, Y_test))
print('Bagging_accuracy score: ',Bagging.score(X_test, Y_test))
print('Ada_boost_accuracy score: ',AdaBoost.score(X_test, Y_test))
print('Gradient_boosting_accuracy score: ',GB.score(X_test, Y_test))

