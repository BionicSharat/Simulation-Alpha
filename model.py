import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import Iterable

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
        return outputs1, outputs2, outputs3

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
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64) 
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        outputs1, outputs2, outputs3 = self.model(state)

        target1 = []
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max([q[0] for q in self.model(next_state[idx])[0]])

            target1.append(Q_new)
    
        target2 = []
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max([q[0] for q in self.model(next_state[idx])[1]])

            target2.append(Q_new)

        target3 = []
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max([q[0] for q in self.model(next_state[idx])[2]])

            target3.append(Q_new)

        # convert to tensor
        target1 = torch.tensor(target1, dtype=torch.float)
        target2 = torch.tensor(target2, dtype=torch.float)
        target3 = torch.tensor(target3, dtype=torch.float)

        # clear gradients
        self.optimizer.zero_grad()
        # calculate loss
        output_tensor1 = torch.cat(outputs1, dim=0)
        output_index1 = torch.argmax(output_tensor1[action[0]])

        output_tensor2 = torch.cat(outputs2, dim=0)
        output_index2 = torch.argmax(output_tensor2[action[0]])

        output_tensor3 = torch.cat(outputs3, dim=0)
        output_index3 = torch.argmax(output_tensor3[action[0]])

        loss1 = self.criterion(output_index1, target1)
        loss2 = self.criterion(output_index2, target2)
        loss3 = self.criterion(output_index3, target3)
        loss = sum(flatten([loss1, loss2, loss3]))

        # backpropagation
        loss.backward()
        self.optimizer.step()

    def flatten(l):
        return [item for sublist in l for item in sublist]