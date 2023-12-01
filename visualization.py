import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set()
list_not_use_discrete = ["name_fuzz_ratio", "description_polarity", 
                         "description_subjectivity"]


def draw_hist(data):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for idx, col in enumerate(data.columns):
        if col == 'label':
            continue
        if col in list_not_use_discrete:
            sns.histplot(data=data, x=col, ax=axes[idx//5, idx%5], 
                        kde=True, stat="probability", hue='label',
                        element="bars", common_norm=False, bins=50)
        else:
            sns.histplot(data=data, x=col, ax=axes[idx//5, idx%5], 
                        kde=True, stat="probability", hue='label',
                        element="bars", common_norm=False, discrete=True)
    plt.tight_layout()
    plt.savefig('figs/distribution.png')
    plt.clf()


def draw_box(data):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for idx, col in enumerate(data.columns):
        if col == 'label':
            continue
        sns.boxplot(y=col, data=data, hue='label', ax=axes[idx//5, idx%5],
                    showfliers=True, width=.5, gap=.2)
    plt.tight_layout()
    plt.savefig('figs/difference.png')
    plt.clf()


def draw_corr(data):
    sns.set_context({"figure.figsize":(10,10)})
    sns.heatmap(data=data.corr(), square=True, cmap='RdBu_r')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('figs/corr.png')
    plt.clf()


def draw_neighbor(data):
    labels = data['label'].values
    data = data.drop(['label'], axis=1).values

    # normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # t-sne
    tsne = TSNE(n_components=2, learning_rate=500, n_iter=10000, metric='l1')
    data = tsne.fit_transform(data) 
    print(data.shape)

    # plot
    cmap = plt.cm.Spectral
    plt.figure(figsize=(10, 10))
    for i in range(2):
        indices = labels == i
        plt.scatter(data[indices, 0], data[indices, 1], color=cmap(i/1.1), label=i)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/neighbor.png')


if __name__ == '__main__':
    with open('data_new/train.csv', 'r') as f:
        train_df = pd.read_csv(f)
    with open('data_new/dev.csv', 'r') as f:
        dev_df = pd.read_csv(f)
    all_df = pd.concat([train_df, dev_df], axis=0, ignore_index=True)

    drop_list = [f'mined_feature_{i}' for i in range(3, 10)]
    all_df.drop(drop_list, axis=1, inplace=True)

    #draw_hist(all_df)
    #draw_box(all_df)
    #draw_corr(all_df)
    draw_neighbor(all_df)
    