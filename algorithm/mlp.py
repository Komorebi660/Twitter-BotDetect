import random
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)                           # python random seed
    np.random.seed(seed)                        # numpy random seed
    os.environ['PYTHONHASHSEED'] = str(seed)    # python hash seed
    torch.manual_seed(seed)                     # pytorch random seed
    torch.cuda.manual_seed(seed)                # cuda random seed


class DenseBlock(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims*3, input_dims*2),
            nn.Linear(input_dims*2, input_dims),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.norm = nn.BatchNorm1d(input_dims) 
    def forward(self, inputs, dense):
        x = self.norm(self.mlp(inputs))
        out = inputs + torch.cat([x, dense], dim=1)
        return out


class DenseModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.norm = nn.BatchNorm1d(input_dims)  

        self.input_up_proj = nn.Sequential(
            nn.Linear(input_dims, int(input_dims*1.5)),
        )

        self.dense_layers = nn.ModuleList([DenseBlock(input_dims//2) for _ in range(6)])

        self.output_down_proj = nn.Sequential(
            nn.Linear(int(input_dims*1.5), input_dims),
            nn.Linear(input_dims, 1),
        )
        
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, labels=None):
        inputs = self.norm(inputs)                      # [batch_size, 32]
        x = self.input_up_proj(inputs)                  # [batch_size, 48]
        for dense in self.dense_layers:
            x = dense(x, inputs)                        # [batch_size, 48]
        prediction_scores = self.output_down_proj(x)    # [batch_size, 1]
        logits = self.sigmoid(prediction_scores)        # [batch_size, 1]

        if labels is None:
            return logits
        else:
            loss = self.BCEloss(prediction_scores, labels.unsqueeze(1).float())
            return logits, loss



class MyDataset(Dataset):
    def __init__(self, split='train'):
        with open(f'../data_new/{split}.csv', 'r') as f:
            dataset = pd.read_csv(f)
        
        self.features = dataset.drop(['label'], axis=1).values.astype(np.float32)
        self.labels = dataset['label'].values.astype(np.float32)
        print(self.features.shape)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        with torch.no_grad():
            data_item = {"inputs": self.features[index], "label": self.labels[index]}
        return data_item


def create_data_loader(split, batch_size):
    dataset = MyDataset(split)

    if split == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    
    return data_loader


def train(args, device):
    train_dataloader = create_data_loader('train', args.bsz)
    dev_dataloader = create_data_loader('dev', None)
    # get full dev dataset
    dev_dataset = []
    for dev_data in dev_dataloader:
        dev_inputs = dev_data['inputs']
        dev_labels = dev_data['label']
        dev_dataset.append((dev_inputs, dev_labels))
    input_dims = dev_dataset[0][0].shape[1]
    print(f"example data: {dev_dataset[0][0]}")
    print(f"input_dims: {input_dims}")

    # get model
    model = DenseModel(input_dims).to(device)
    model.train()
    # get optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # learning rate coefficient for each step
    def lr_lambda(step):
        fraction = step / (args.epochs * len(train_dataloader))
        return 1 - fraction
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # train & validate on-the-fly
    training_losses = []
    training_accuracy = []
    dev_losses = []
    dev_accuracy = []
    for epoch in trange(args.epochs):
        #print(f"Epoch {epoch+1} ...")
        # training
        all_loss = []
        all_acc = []
        for data in train_dataloader:
            inputs = data['inputs'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            prediction_scores, loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            all_loss.append(loss.detach().cpu().numpy())
            prediction = (prediction_scores > 0.5).int().cpu().numpy()
            all_acc.append(f1_score(labels.cpu().numpy(), prediction, zero_division=1, average='micro'))
            scheduler.step()
        _loss = sum(all_loss)/len(all_loss)
        training_losses.append(_loss)
        _acc = sum(all_acc)/len(all_acc)
        training_accuracy.append(_acc)
        #print(f"training Loss: {_loss}")
        #print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        #print(f"training accuracy: {_acc}")
        
        # evaluation
        model.eval()
        with torch.no_grad():
            all_dec_acc = []
            all_dev_loss = []
            for dev_inputs, dev_labels in dev_dataset:
                dev_inputs = dev_inputs.to(device)
                dev_labels = dev_labels.to(device)
                dev_prediction_scores, dev_loss = model(dev_inputs, dev_labels)
                dev_prediction = (dev_prediction_scores > 0.5).int().cpu().numpy()
                all_dec_acc.append(f1_score(dev_labels.cpu().numpy(), dev_prediction, zero_division=1, average='micro'))
                all_dev_loss.append(dev_loss.cpu().numpy())
            dev_loss = sum(all_dev_loss)/len(all_dev_loss)
            dev_losses.append(dev_loss)
            dev_acc = sum(all_dec_acc)/len(all_dec_acc)
            dev_accuracy.append(dev_acc)
            #print(f"dev Loss: {dev_loss}")
            #print(f"dev accuracy: {dev_acc}")
        model.train()
    
    # plot loss curve
    #plt.ylim(0.30, 0.35)
    plt.plot(training_losses, label='training loss')
    plt.plot(dev_losses, label='dev loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()
    # plot accuracy curve
    plt.ylim(0.70, 0.76)
    plt.plot(training_accuracy, label='training accuracy', linestyle='--')
    plt.plot(dev_accuracy, label='dev accuracy')
    print(f"final dev accuracy: {dev_accuracy[-1]}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.close()
    
    return model.eval()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1986, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="current process rank")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")    
    args = parser.parse_args()

    set_seed(2023)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = train(args, device)
   