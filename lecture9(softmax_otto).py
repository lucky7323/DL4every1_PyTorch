import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def preprocessing():
    df = pd.read_csv('./data/otto/train.csv')
    f = lambda x: int(x[6:])
    df['target'] = df['target'].apply(f)
    df.to_csv('./data/otto/train_processed.csv', index=False, header=False)

class Dataset(Dataset):
    def __init__(self):
        preprocessing()
        xy = np.loadtxt('./data/otto/train_processed.csv', delimiter=',', skiprows=1)
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = len(xy)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def __len__(self):
        return self.len

dataset = Dataset()
