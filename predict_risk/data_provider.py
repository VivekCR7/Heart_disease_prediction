import os
import pickle
import joblib

config = {
    'heart': {
        'RandomForest': 'production/RF.pkl',
        'KNN': 'production/KNN.pkl',
        'NaiveBayes': 'production/naive_bayes.pkl',
        'DecisionTree':'production/decision_tree.pkl',
        'scalar_file': 'production/standard_scalar.pkl',
    }}

dir = os.path.dirname(__file__)

def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    else:
        print("file does not exit")

def GetPickleFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return pickle.load( open(os.path.join(dir, filepath), "rb" ) )
    return None

def GetStandardScalarForHeart():
    return GetPickleFile(config['heart']['scalar_file'])

def GetAllClassifiersForHeart():
    return (GetRFClassifierForHeart(),GetKnnClassifierForHeart(),GetNaiveBayesClassifierForHeart(),GetDecisionTreeClassifierForHeart())

def GetRFClassifierForHeart():
    return GetJobLibFile(config['heart']['RandomForest'])

def GetKnnClassifierForHeart():
    return GetJobLibFile(config['heart']['KNN'])

def GetNaiveBayesClassifierForHeart():
    return GetJobLibFile(config['heart']['NaiveBayes'])

def GetDecisionTreeClassifierForHeart():
    return GetJobLibFile(config['heart']['DecisionTree'])
