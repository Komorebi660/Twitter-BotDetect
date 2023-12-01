from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pandas as pd
import argparse

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def load_data(split):
    with open(f'../data_new/{split}.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['label'], axis=1).values
    labels = dataset['label'].values
    return features, labels


def gridsearch(classifier, param_grid):
    # read data
    X_train , Y_train = load_data('train')

    # grid search
    grid_search = GridSearchCV(classifier, param_grid, scoring='f1', \
                               cv=5, n_jobs=16, verbose=1)
    grid_search.fit(X_train, Y_train)
    print(f'Best Grid parameters On Train: {grid_search.best_params_}')
    print(f'Best Grid score On Train: {grid_search.best_score_}')

    return grid_search.best_params_


def run(classifier):
    # Load training data
    X_train, Y_train = load_data('train')
    print(X_train.shape, Y_train.shape)
    
    classifier.fit(X_train, Y_train)
    Y_train_pred = classifier.predict(X_train)
    print('On train set:')
    print(f1_score(Y_train, Y_train_pred, zero_division=1, average='macro'))
    #print(f1_score(Y_train, Y_train_pred, zero_division=1, average='micro'))
    
    X_dev, Y_dev = load_data('dev')
    Y_dev_pred = classifier.predict(X_dev)
    print('On dev set:')
    print(f1_score(Y_dev, Y_dev_pred, zero_division=1, average='macro'))
    #print(f1_score(Y_dev, Y_dev_pred, zero_division=1, average='micro'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='algorithm name')
    args = parser.parse_args()

    if args.algo == 'svm':
        classifier = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 0.2, 0.5, 0.8, 1, 2],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 0.8, 'kernel': 'linear', 'random_state': 42}
        run(SVC(**best_params_))
    elif args.algo == 'xgb':
        classifier = XGBClassifier(objective='binary:logistic', \
                                subsample=0.8, colsample_bytree=0.8, \
                                random_state=42, n_jobs=16)
        param_grid = {
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [500, 800, 1000, 1200, 1500, 2000],
            'min_child_weight': [0.5, 0.7, 0.9, 1, 1.1],
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'learning_rate': 0.01, 'n_estimators': 1000, \
                        'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.9, \
                        'reg_alpha': 0.1, 'reg_lambda': 0.5, \
                        'objective': 'binary:logistic', \
                        'subsample': 0.8, 'colsample_bytree': 0.8, \
                        'random_state': 42, 'n_jobs': 16}
        run(XGBClassifier(**best_params_))
    elif args.algo == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [i for i in range(2, 21, 2)],
            'min_samples_split': [i for i in range(2, 22, 2)],
            'min_samples_leaf': [i for i in range(1, 11, 1)],
            'n_estimators': [i for i in range(50, 151, 10)],
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                       'max_features': 'sqrt', 'n_estimators': 150, \
                       'min_samples_leaf': 5, 'min_samples_split': 10, \
                       'random_state': 42}
        run(RandomForestClassifier(**best_params_))
    elif args.algo == 'nb':
        classifier = GaussianNB()
        run(classifier)
    elif args.algo == 'lr':
        classifier = LogisticRegression(random_state=42, n_jobs=16)
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.1, 0.2, 0.5, 0.8, 1, 2],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [500, 1000, 2000, 3000]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 2, 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga', \
                        'random_state': 42, 'n_jobs': 16}
        run(LogisticRegression(**best_params_))
    else:
        print('Invalid algorithm name!')