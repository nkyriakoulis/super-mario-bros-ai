import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# PPO algorithm https://arxiv.org/pdf/1707.06347.pdf

# The input is four frames (images). We will use a hybrid NN consisting of a CNN, to efficiently capture
# information in the images, get a vector as output and feed it to a Q Network of 3 layers, similar to
# the snake game.
# We will use the pre-trained AlexNet network as our CNN, that will output a 64x1 vector.
# This vector will be fed to a 64, 256, 7 fully connected layer, as we have 7 possible moves

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)

# Freeze the pre-trained layers
for param in alexnet.parameters():
    param.requires_grad = False

# Modify the last fully connected layer
num_features = alexnet.classifier[-1].in_features  # Get the number of input features of the last FC layer
alexnet.classifier[-1] = nn.Linear(num_features, 16)  # Change the last FC layer to output a 4x16=64 matrix


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Assuming 7 output units for classification

    def forward(self, x):
        alexnet_output = alexnet(x[0])

        x = F.relu(self.fc1(alexnet_output.view(1, -1)))
        x = self.fc2(x)

        return x

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
        action = torch.tensor(action, dtype=torch.int)
        reward = torch.tensor(reward, dtype=torch.float)

        if reward.ndimension() == 0:
            # If it is a single number, reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for i in range(len(state)):
            Q_new = reward[i]
            if not done:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
