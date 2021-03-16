import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv("Heart_disease.csv")

data['disease'] = np.where(data['diagnosis']>0,1,0)
data.rename(columns={'diagnosis':'diag'},inplace= True)

if data.isnull().to_numpy().any():
    data = data.fillna(data.median())

#check for the duplicate rows
duplicated = data.duplicated().sum()
if duplicated:
    print('Duplicates Rows in DataSet are: {}'.format(duplicated))
else:
    print('Dataset contains no Duplicate Values')

print()


#function to list all the outliers
continous_features = ['age','restbp','chol','maxhr','oldpeak']
def outliers(data_out, drop = False):
    for each_features in data_out.columns:
        feature_data = data_out[each_features]
        q1 = np.percentile(feature_data, 25.)
        q3 = np.percentile(feature_data, 75.)
        
        IQR = q3-q1 #interquartile range
        
        outlier_step = IQR * 1.5
        
        outliers = feature_data[~((feature_data >= q1-outlier_step) & (feature_data <= q3 + outlier_step))].index.tolist()
        
        if not drop:
            print('For the feature {}, No. of outliers is {}'.format(each_features, len(outliers)))
        
        if drop:
            data.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_features))

outliers(data[continous_features])

print()

#drop the outliers
outliers(data[continous_features], drop = True)
print()

#importing machine learning libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier     
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

#import other dependent libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image
from sklearn import metrics
from io import StringIO
import pydotplus

#split data setn into train and test data

features = ['age','sex','cp','restbp','chol','fbs','restecg','maxhr','exang','oldpeak','slope','mv','thal']

X= data[features]
Y= data['disease']

#splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#model fitting

#1) Decision Tree

dtree_1 = tree.DecisionTreeClassifier()
dtree_1.fit(X_train,Y_train)
y_pred = dtree_1.predict(X_test)

print('**********************  Decision Tree *************************')

print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

accuracy = metrics.accuracy_score(Y_test, y_pred)
print('Accuracy for Decision Tree Classifier is : {:.2f}'.format(accuracy))
print()

#create png image of descion tree for visualization
dot_data = StringIO()
export_graphviz(dtree_1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decisiontree.png')
Image(graph.create_png())

#2) Random Forest

model = RandomForestClassifier().fit(X_train,Y_train)
Y_pred_rf = model.predict(X_test)

print('**********************  Random Forest Classifier *************************')

print(confusion_matrix(Y_test, Y_pred_rf))
print(classification_report(Y_test, Y_pred_rf))

accuracy = metrics.accuracy_score(Y_test, Y_pred_rf)
print('Accuracy for Random Forest Classifier is: {:.2f}'.format(accuracy))
print()

#3) Naive Bayes

nb = GaussianNB()
Y_train = Y_train.astype('int')

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)

print('********************** Naive Bayes *************************')

print(confusion_matrix(Y_test, Y_pred_nb))
print(classification_report(Y_test, Y_pred_nb))

accuracy = metrics.accuracy_score(Y_test, Y_pred_nb)
print('Accuracy for Naive Bayes: {:.2f}'.format(accuracy))
print()

#4) KNN classifier


"""
To make this algorithm more accurate feature scaling is needed to scale the features 
so they can be evaluated uniformly. To normalize the range of all the features 
I used the StandardScaler method to transform the data so the distribution 
will have a mean value 0 and standard deviation of 1."""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,Y_train)
Y_pred_knn = knn.predict(X_test)

print('**********************  KNN classifier *************************')

print(confusion_matrix(Y_test, Y_pred_knn))
print(classification_report(Y_test, Y_pred_knn))

accuracy = metrics.accuracy_score(Y_test, Y_pred_knn)
print('Accuracy for KNN classifier: {:.2f}'.format(accuracy))

