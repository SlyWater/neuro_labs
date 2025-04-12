import numpy as np
import pandas as pd
import torch
from torch import nn, optim

df = pd.read_csv('C:/Users/Vadim/PycharmProjects/neuro/2_Основы python_обучение нейрона/data.csv')

y = df.iloc[:, 4].values
y = torch.tensor(np.where(y == "Iris-setosa", 1, -1), dtype=torch.float32)
X = torch.tensor(df.iloc[:, 0:3].values, dtype=torch.float32)

linear = nn.Linear(3, 1)

w = torch.randn(4)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(linear.parameters(), lr=0.01)

for epoch in range(1000):
    pred = linear(X).squeeze()

    loss = loss_fn(pred, y)
    print(f'Ошибка на итерации {epoch}: {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
