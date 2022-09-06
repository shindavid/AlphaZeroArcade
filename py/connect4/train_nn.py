#!/usr/bin/env python3
import argparse
from typing import Tuple

import h5py
import itertools

import numpy as np
from natsort import natsorted
import os
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


Shape = Tuple[int, ...]


class Net(nn.Module):
    def __init__(self, input_shape: Shape, n_conv_filters=64):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(input_shape[0], n_conv_filters, 2)  # 64 x 6 x 5
        self.batch = nn.BatchNorm2d(n_conv_filters)

        # TODO

        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 16, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 3 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3a = nn.Linear(84, 7)
        # self.fc3b = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 1)  # (4, 5, 4)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 1)  # (16, 3, 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        ya = self.fc3a(x)
        return ya
        # yb = self.fc3b(x)
        # return ya, yb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--games-dir", default="c4_games")

    return parser.parse_args()


def main():
    args = get_args()

    games_dir = args.games_dir
    assert os.path.isdir(games_dir)

    full_input_data = []
    full_policy_output_data = []
    full_value_output_data = []

    print('Loading data...')
    for filename in natsorted(os.listdir(games_dir)):
        full_filename = os.path.join(games_dir, filename)
        with h5py.File(full_filename, 'r') as f:
            input_data = f['input'][()]
            policy_output_data = f['policy_output'][()]
            value_output_data = f['value_output'][()]
            full_input_data.append(input_data)
            full_policy_output_data.append(policy_output_data)
            full_value_output_data.append(value_output_data)
    print('Data loaded!')

    full_input_data = np.concatenate(full_input_data)
    full_policy_output_data = np.concatenate(full_policy_output_data)
    full_value_output_data = np.concatenate(full_value_output_data)

    print('input', full_input_data.shape)
    print('policy', full_policy_output_data.shape)
    print('value', full_value_output_data.shape)
    if True:
        return


    net = Net()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Loading data...')
    with open('output.pkl', 'rb') as f:
        full_data = pickle.load(f)
    print('Data loaded!')


    train_pct = 0.9


    random.shuffle(full_data)
    num_games = len(full_data)
    num_train_games = int(num_games * train_pct)
    train_data = list(itertools.chain(*full_data[:num_train_games]))
    test_data = list(itertools.chain(*full_data[num_train_games:]))


    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return self.data[item]


    batch_size = 64
    train_loader = DataLoader(CustomDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CustomDataset(test_data), batch_size=len(test_data))


    num_epochs = 4
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, value_label, policy_label = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, policy_label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # N = 100
            # if i % N == (N-1):
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / N:.3f}')
            #     running_loss = 0.0

        for data in test_loader:
            inputs, value_label, policy_label = data
            outputs = net(inputs)
            loss = criterion(outputs, policy_label)
            avg_test_loss = loss.item()
            predicted_best_moves = torch.argmax(outputs, axis=1)

            correct = policy_label.gather(1, predicted_best_moves.view(-1, 1))
            accuracy = float(sum(correct)) / len(correct)
            min_move = min(predicted_best_moves)
            max_move = max(predicted_best_moves)
            print(f'Epoch {epoch} ended! Avg test loss: {avg_test_loss:.3f} Accuracy: {100*accuracy:.3f}% move_range:[{min_move}, {max_move}]')

    print('Finished Training')



    #
    # torch.manual_seed(123)
    #
    # net = Net()
    # print(net)
    #
    # params = list(net.parameters())
    # print(len(params))
    # print(params[0].size())  # conv1's .weight
    #
    # input = torch.randn(1, 1, 32, 32)
    # out = net(input)
    # print(out)
    #
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))
    #
    # output = net(input)
    # target = torch.randn(10)  # a dummy target, for example
    # target = target.view(1, -1)  # make it the same shape as output
    # criterion = nn.MSELoss()
    #
    # loss = criterion(output, target)
    # print(loss)
    #
    # net.zero_grad()     # zeroes the gradient buffers of all parameters
    #
    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)
    #
    # loss.backward()
    #
    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)
    #
    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)


if __name__ == '__main__':
    main()
