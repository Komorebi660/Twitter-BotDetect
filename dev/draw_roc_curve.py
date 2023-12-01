from sklearn.metrics import roc_curve
import pandas as pd

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


def load_data(split):
    with open(f'../data_new/{split}.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['label'], axis=1).values
    labels = dataset['label'].values
    return features, labels


def run(classifier):
    # Load training data
    X_train, Y_train = load_data('train')
    print(X_train.shape, Y_train.shape)
    
    classifier.fit(X_train, Y_train)
    
    X_dev, Y_dev = load_data('dev')
    try:
        Y_dev_pred = classifier.predict_proba(X_dev)[:, 1]
    except:
        try:
            Y_dev_pred = classifier.decision_function(X_dev)
        except:
            raise ValueError('classifier not support predict_proba or decision_function')

    fpr, tpr, _ = roc_curve(Y_dev, Y_dev_pred)
    
    return fpr, tpr


def get_data():
    data = {}
    _best_params_ = {'C': 0.8, 'kernel': 'linear', 
                     'random_state': 42, 'probability': True}
    fpr, tpr = run(SVC(**_best_params_))
    data['svm'] = (fpr, tpr)

    best_params_ = {'learning_rate': 0.01, 'n_estimators': 1000, \
                    'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.9, \
                    'reg_alpha': 0.1, 'reg_lambda': 0.5, \
                    'objective': 'binary:logistic', \
                    'subsample': 0.8, 'colsample_bytree': 0.8, \
                    'random_state': 42, 'n_jobs': 16}
    fpr, tpr = run(XGBClassifier(**best_params_))
    data['xgboost'] = (fpr, tpr)

    best_param_ = {'criterion': 'gini', 'max_depth': 10, \
                    'max_features': 'sqrt', 'n_estimators': 150, \
                    'min_samples_leaf': 5, 'min_samples_split': 10, \
                    'random_state': 42}
    fpr, tpr = run(RandomForestClassifier(**best_param_))
    data['random forest'] = (fpr, tpr)

    fpr, tpr = run(GaussianNB())
    data['naive bayes'] = (fpr, tpr)

    best_params_ = {'C': 2, 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga', \
                    'random_state': 42, 'n_jobs': 16}
    fpr, tpr = run(LogisticRegression(**best_params_))
    data['logistic regression'] = (fpr, tpr)

    return data


def draw_roc_curve():
    data = get_data()
    for key in data:
        fpr, tpr = data[key]
        plt.plot(fpr, tpr, label=key)
    #plt.xlim([0.2, 0.4])
    #plt.ylim([0.7, 0.9])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/roc_curve.png')


if __name__ == '__main__':
    draw_roc_curve()
