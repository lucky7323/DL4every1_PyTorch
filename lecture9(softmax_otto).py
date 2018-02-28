import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

batch_size = 128

def preprocessing():
    df = pd.read_csv('./data/otto/train.csv')
    f = lambda x: int(x[6:]) - 1
    df['target'] = df['target'].apply(f)
    df.to_csv('./data/otto/train_processed.csv', index=False, header=False)

class Dataset(Dataset):
    def __init__(self):
        preprocessing()
        xy = np.loadtxt('./data/otto/train_processed.csv',
                        delimiter=',', skiprows=1, dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, 1:-1])
        y = torch.from_numpy(xy[:, [-1]])
        self.y_data = torch.squeeze(y.type(torch.LongTensor))
        self.len = len(xy)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def __len__(self):
        return self.len

dataset = Dataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(93, 50)
        self.l2 = torch.nn.Linear(50, 20)
        self.l3 = torch.nn.Linear(20, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

for epoch in range(1, 11):
    train(epoch)
