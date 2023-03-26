import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import Iterable
import numpy as np

def flatten(items):
    for elem in items:
        if isinstance(elem,Iterable) and not isinstance(elem,str):
            for sub_elem in flatten(elem):
                yield sub_elem
        else:
            yield elem

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.outputs1 = nn.ModuleList([nn.Linear(128, 100) for _ in range(output_sizes[0])])
        self.outputs2 = nn.ModuleList([nn.Linear(128, 7) for _ in range(output_sizes[1])])
        self.outputs3 = nn.ModuleList([nn.Linear(128, 2) for _ in range(output_sizes[2])])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        outputs1 = [output(x) for output in self.outputs1]
        outputs2 = [output(x) for output in self.outputs2]
        outputs3 = [output(x) for output in self.outputs3]
        return [outputs1, outputs2, outputs3]

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.int64) 
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        pred1 = pred[0]
        pred2 = pred[1]
        pred3 = pred[2]

        losses1 = []
        for i in range(7):
            target = pred1[i].clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])[0][i])
                target[idx][torch.argmax(action[idx][i]).item()] = Q_new
            losses1.append(self.criterion1(target, pred1[i]))
    
        losses2 = []
        for i in range(14):
            target = pred2[i].clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])[1][i])
                target[idx][torch.argmax(action[idx][i+7]).item()] = Q_new
            losses2.append(self.criterion2(target, pred2[i]))

        losses3 = []
        for i in range(7):
            target = pred3[i].clone()
            
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])[2][i])
                target[idx][torch.argmax(action[idx][i+21]).item()] = Q_new
            losses3.append(self.criterion3(target, pred3[i]))

        # clear gradients
        self.optimizer.zero_grad()
        # calculate loss
        loss1 = sum(losses1)
        loss2 = sum(losses2)
        loss3 = sum(losses3)
        # backpropagation
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward(retain_graph=True)
        self.optimizer.step()