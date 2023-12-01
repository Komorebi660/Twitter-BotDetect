from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import json
import numpy as np
import pandas as pd


def load_data(split):
    with open(f'./data_new/{split}.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['label'], axis=1).values
    labels = dataset['label'].values
    return features, labels


if __name__ == '__main__':
    # Load training data
    X_1, Y_1 = load_data('train')
    #print(X_1.shape, Y_1.shape)
    X_2, Y_2 = load_data('dev')
    #print(X_2.shape, Y_2.shape)
    X_train = np.concatenate((X_1, X_2), axis=0)
    Y_train = np.concatenate((Y_1, Y_2), axis=0)
    print(X_train.shape, Y_train.shape)
    
    # Initialize XGBoost classifier
    classifier = XGBClassifier(objective='binary:logistic', \
                                learning_rate=0.01, max_depth=5, n_estimators=1000, \
                                subsample=0.8, colsample_bytree=0.8, \
                                min_child_weight=0.9, gamma=0.05, \
                                reg_alpha=0.1, reg_lambda=0.5, \
                                random_state=42, n_jobs=16)
    
    # training
    classifier.fit(X_train, Y_train)
    Y_train_pred = classifier.predict(X_train)
    print('On train set:')
    print(f1_score(Y_train, Y_train_pred, zero_division=1, average='macro'))
    print(f1_score(Y_train, Y_train_pred, zero_division=1, average='micro'))
    
    X_test, _ = load_data('test')
    Y_test_pred = classifier.predict(X_test)

    with open('./data/test.json', 'r') as f:
        dataset = json.load(f)
    results = []
    for i, user_data in enumerate(dataset):
        user_data['label'] = 'bot' if int(Y_test_pred[i])==1 else 'human'
        results.append(user_data)
    with open('./test_label.json', 'w') as f:
        json.dump(results, f, indent=4)
