#!/usr/bin/env python3
import argparse
import os
import random
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from natsort import natsorted
from torch.utils.data import DataLoader

from game import NUM_COLUMNS, NUM_ROWS


Shape = Tuple[int, ...]


class ConvBlock(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The convolutional block applies the following modules:

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_input_channels: int, n_conv_filters: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, n_conv_filters, kernel_size=3, stride=1, padding=1)
        self.batch = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        return F.relu(self.batch(self.conv(x)))


class ResBlock(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    Each residual block applies the following modules sequentially to its input:

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_conv_filters: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(n_conv_filters)
        self.conv2 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        identity = x
        x = F.relu(self.batch1(self.conv1(x)))
        x = self.batch2(self.conv2(x))
        x += identity  # skip connection
        return F.relu(x)


class PolicyHead(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The policy head applies the following modules:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19^2 + 1 = 362 corresponding to
    logit probabilities for all intersections and the pass move

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self, n_input_channels: int):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, 2, kernel_size=1, stride=1)
        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2 * NUM_COLUMNS * NUM_ROWS, NUM_COLUMNS)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, 2 * NUM_COLUMNS * NUM_ROWS)
        x = self.linear(x)
        return x
        # return self.linear(F.relu(self.batch(self.conv(x))).view(2 * NUM_COLUMNS * NUM_ROWS))


class ValueHead(nn.Module):
    """
    From "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero paper):

    The value head applies the following modules:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [−1, 1]

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    def __init__(self):
        super(ValueHead, self).__init__()
        raise Exception('TODO')

    def forward(self, x):
        pass


class Net(nn.Module):
    def __init__(self, input_shape: Shape, n_conv_filters=64, n_res_blocks=9):
        super(Net, self).__init__()
        self.conv_block = ConvBlock(input_shape[0], n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x)


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

    full_data = list(zip(full_input_data, full_value_output_data, full_policy_output_data))

    net = Net(full_input_data[0].shape)
    net.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    train_pct = 0.9

    random.shuffle(full_data)
    train_n = int(len(full_data) * train_pct)
    train_data = full_data[:train_n]
    test_data = full_data[train_n:]

    batch_size = 256
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), pin_memory=True)

    num_epochs = 8
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, value_label, policy_label = data
            inputs = inputs.to('cuda', non_blocking=True)
            value_label = value_label.to('cuda', non_blocking=True)
            policy_label = policy_label.to('cuda', non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, policy_label)
            loss.backward()
            optimizer.step()

        for data in test_loader:
            inputs, value_label, policy_label = data
            inputs = inputs.to('cuda', non_blocking=True)
            value_label = value_label.to('cuda', non_blocking=True)
            policy_label = policy_label.to('cuda', non_blocking=True)
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


if __name__ == '__main__':
    main()
