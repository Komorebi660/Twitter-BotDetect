import pandas as pd

from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance

import matplotlib.pyplot as plt


def load_data():
    with open(f'../data_new/train.csv', 'r') as f:
        train_df = pd.read_csv(f)
    with open(f'../data_new/dev.csv', 'r') as f:
        dev_df = pd.read_csv(f)
    dataset = pd.concat([train_df, dev_df], axis=0, ignore_index=True)
    columns = [col for col in dataset.drop(['label'], axis=1).columns]
    features = dataset.drop(['label'], axis=1).values
    labels = dataset['label'].values
    return features, labels, columns


def run():
    # Load training data
    features, labels, columns = load_data()
    params_ = {'learning_rate': 0.01, 'n_estimators': 1000, \
                'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.9, \
                'reg_alpha': 0.1, 'reg_lambda': 0.5, \
                'subsample': 0.8, 'colsample_bytree': 0.8, \
                'objective': 'binary:logistic', \
                'random_state': 42, 'n_jobs': 16 }
    classifier = XGBClassifier(**params_)
    classifier.fit(features, labels)
    classifier.get_booster().feature_names = columns
    return classifier


def _plot_importance(classifier):
    plot_importance(classifier, max_num_features=32, 
                    xlabel='# of occurrences in trees',
                    title='', importance_type='weight',
                    values_format='{v:.0f}', xlim=(0,2500))
    plt.tight_layout()
    plt.savefig('../figs/xgboost_feature_weight.png')
    plt.clf()

    plot_importance(classifier, max_num_features=32, 
                    xlabel='performance gain',
                    title='', importance_type='gain',
                    values_format='{v:.2f}', xlim=(0,30))
    plt.tight_layout()
    plt.savefig('../figs/xgboost_feature_gain.png')
    plt.clf()


def _plot_tree(classifier, i):
    plot_tree(classifier, num_trees=i-1)
    fig = plt.gcf()
    fig.set_size_inches(50, 6)
    plt.tight_layout()
    plt.savefig(f'../figs/tree_{i}.png')
    plt.clf()


if __name__ == '__main__':
    classifier = run()
    #_plot_importance(classifier)
    for i in [1, 10, 100, 200, 500, 1000]:
        _plot_tree(classifier, i)
    