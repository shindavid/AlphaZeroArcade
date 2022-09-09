#!/usr/bin/env python3
import argparse
import copy
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset

from neural_net import Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--games-dir", default="c4_games", help='c4 games dir (default: %(default)s)')
    parser.add_argument("-m", "--model-file", default="c4_model.pt",
                        help='model output location (default: %(default)s)')
    parser.add_argument("-w", "--weak-mode", action='store_true', help='Weak mode (default: strong)')
    parser.add_argument("-e", "--num-epochs", type=int, default=16, help='Num epochs (default: %(default)s)')
    parser.add_argument("-b", "--batch-size", type=int, default=64, help='Batch size (default: %(default)s)')
    parser.add_argument("-r", "--num-residual-blocks", type=int, default=19,
                        help='Num residual blocks (default: %(default)s)')

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def main():
    args = get_args()
    if args.weak_mode:
        raise Exception('TODO: figure out how to handle weak mode better. Exclude all data in losing positions?')

    games_dir = args.games_dir
    assert os.path.isdir(games_dir)

    full_input_data = []
    full_policy_output_data = []
    full_value_output_data = []

    policy_key = 'weak_policy' if args.weak_mode else 'strong_policy'
    print('Loading data...')
    for filename in natsorted(os.listdir(games_dir)):
        full_filename = os.path.join(games_dir, filename)
        with h5py.File(full_filename, 'r') as f:
            input_data = f['input'][()]
            policy_output_data = f[policy_key][()]
            value_output_data = f['value'][()]
            full_input_data.append(input_data)
            full_policy_output_data.append(policy_output_data)
            full_value_output_data.append(value_output_data)

    full_input_data = np.concatenate(full_input_data)
    full_policy_output_data = np.concatenate(full_policy_output_data)
    full_value_output_data = np.concatenate(full_value_output_data)

    full_data = list(zip(full_input_data, full_value_output_data, full_policy_output_data))
    print(f'Data loaded! Num positions: {len(full_data)}')

    net = Net(full_input_data[0].shape, n_res_blocks=args.num_residual_blocks)
    net.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()

    train_pct = 0.9

    random.shuffle(full_data)
    train_n = int(len(full_data) * train_pct)
    train_data = full_data[:train_n]
    test_data = full_data[train_n:]

    batch_size = args.batch_size
    train_loader = DataLoader(CustomDataset(train_data), batch_size=batch_size, shuffle=True)  # , pin_memory=True)
    test_loader = DataLoader(CustomDataset(test_data), batch_size=len(test_data))  # , pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    num_epochs = args.num_epochs
    scheduler = LambdaLR(optimizer, lambda epoch: 0.1 if epoch*2<num_epochs else .01)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_net = None
    best_test_accuracy = 0

    for epoch in range(num_epochs):
        train_accuracy_num = 0.0
        train_loss_num = 0.0
        train_den = 0
        for i, data in enumerate(train_loader):
            inputs, value_label, policy_label = data
            assert isinstance(inputs, Tensor)
            inputs = inputs.to('cuda')  # , non_blocking=True)

            net.train()
            optimizer.zero_grad()
            outputs = net(inputs)

            # value_label = value_label.to('cuda', non_blocking=True)
            policy_label = policy_label.to('cuda')  # , non_blocking=True)
            loss = criterion(outputs, policy_label)
            n = len(inputs)
            train_loss_num += float(loss.item()) * n
            train_den += n

            selected_moves = torch.argmax(outputs, axis=1)
            correct = policy_label.gather(1, selected_moves.view(-1, 1))
            train_accuracy_num += float(sum(correct))

            loss.backward()
            optimizer.step()

        scheduler.step()
        train_accuracy = train_accuracy_num / train_den
        avg_train_loss = train_loss_num / train_den

        with torch.set_grad_enabled(False):
            for data in test_loader:
                inputs, value_label, policy_label = data
                inputs = inputs.to('cuda', non_blocking=True)

                net.eval()
                outputs = net(inputs)

                # value_label = value_label.to('cuda', non_blocking=True)
                policy_label = policy_label.to('cuda', non_blocking=True)
                loss = criterion(outputs, policy_label)
                avg_test_loss = loss.item()
                selected_moves = torch.argmax(outputs, axis=1)
                correct = policy_label.gather(1, selected_moves.view(-1, 1))
                test_accuracy = float(sum(correct)) / len(correct)
                best = test_accuracy > best_test_accuracy
                print(f'Epoch {epoch} ended! Train loss/accuracy: {avg_train_loss:.3f}/{100*train_accuracy:.3f}% ' +
                      f'Test loss/accuracy: {avg_test_loss:.3f}/{100*test_accuracy:.3f}% {"*" if best else ""}')

                if best:
                    best_net = copy.deepcopy(net)
                    best_test_accuracy = test_accuracy

    print('Finished Training')
    output_dir = os.path.split(args.model_file)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model.constructor_args': best_net.constructor_args,
        'model.state_dict': best_net.state_dict(),
    }, args.model_file)
    print(f'Model saved to {args.model_file}')


if __name__ == '__main__':
    main()
