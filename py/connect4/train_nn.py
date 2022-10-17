#!/usr/bin/env python3
import argparse
import copy
import os
import random
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from connect4.neural_net import Net


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


def get_num_correct_policy_predictions(policy_outputs, policy_labels):
    selected_moves = torch.argmax(policy_outputs, dim=1)
    correct_policy_preds = policy_labels.gather(1, selected_moves.view(-1, 1))
    return int(sum(correct_policy_preds))


def get_num_correct_value_predictions(value_outputs, value_labels):
    value_output_probs = value_outputs.softmax(dim=1)
    deltas = abs(value_output_probs - value_labels)
    return int(sum((deltas < 0.25).all(dim=1)))


def load_data(args):
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
    return full_data, full_input_data[0].shape


class C4DataLoader:
    def __init__(self, args):
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

        train_pct = 0.9

        random.shuffle(full_data)
        train_n = int(len(full_data) * train_pct)
        train_data = full_data[:train_n]
        test_data = full_data[train_n:]

        batch_size = args.batch_size
        train_loader = DataLoader(CustomDataset(train_data), batch_size=batch_size, shuffle=True)  # , pin_memory=True)
        test_loader = DataLoader(CustomDataset(test_data), batch_size=len(test_data))  # , pin_memory=True)

        self.train_n = train_n
        self.input_shape = full_input_data[0].shape
        self.train_loader = train_loader
        self.test_loader = test_loader


def center_text(text: str, n: int) -> str:
    m = n - len(text)
    assert m>=0
    a = m // 2
    b = m - a
    return (' ' * a) + text + (' ' * b)


def main():
    args = get_args()
    if args.weak_mode:
        raise Exception('TODO: figure out how to handle weak mode better. Exclude all data in losing positions?')

    c4_data_loader = C4DataLoader(args)

    net = Net(c4_data_loader.input_shape, n_res_blocks=args.num_residual_blocks)
    net.cuda()

    """
    Normally for AlphaZero, cross-entropy-loss should be used for policy loss. For our current Connect4
    experiment, there are multiple correct moves for a given situation, so we are using
    MultiLabelSoftMarginLoss.
    """
    policy_criterion = nn.MultiLabelSoftMarginLoss()
    value_criterion = nn.CrossEntropyLoss()

    """
    Some constants:

    * value_loss_lambda: used to specify how much weight to place on the value loss, as opposed to the policy loss.
                         KataGo uses 1.5.

    * weight_decay: for L2 regularization. KataGo uses 3e-5 for the L2 regularization constant, which should translate
                    to 2*3e-5 = 6e-5 in weight_decay terms.

    * learning_rate{1,2}: learning rate for the {first, second} half of training. KataGo used a more complex learning
                          rate schedule that we won't try to replicate initially.

    * momentum: KataGo used 0.9

    Regarding value_loss_lambda, note that the AlphaGo Zero paper has the following, which suggests the constant
    should be "small":

    By using a combined policy and value network architecture, and by using a low weight on the
    value component, it was possible to avoid overfitting to the values (a problem described in
    prior work).

    https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
    """
    value_loss_lambda = 0.1
    weight_decay = 6e-5
    learning_rate1 = 0.1
    learning_rate2 = 0.01
    momentum = 0.9

    optimizer = optim.SGD(net.parameters(), lr=learning_rate1, momentum=momentum, weight_decay=weight_decay)
    num_epochs = args.num_epochs
    scheduler = LambdaLR(optimizer, lambda epoch: learning_rate1 if epoch*2<num_epochs else learning_rate2)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_net = None
    best_test_policy_accuracy = 0

    print('%5s %s %s' % ('', center_text('Train', 23), center_text('Test', 23)))
    print('%5s %s %s %s %s' % (
        '', center_text('Value', 11), center_text('Policy', 11),
        center_text('Value', 11), center_text('Policy', 11),
    ))
    print('%5s %5s %5s %5s %5s %5s %5s %5s %5s' % (
        'Epoch', 'Loss', 'Acc', 'Loss', 'Acc', 'Loss', 'Acc', 'Loss', 'Acc',
    ))

    t1 = time.time()
    for epoch in range(num_epochs):
        train_policy_accuracy_num = 0.0
        train_policy_loss_num = 0.0
        train_value_accuracy_num = 0.0
        train_value_loss_num = 0.0
        train_den = 0

        for i, data in enumerate(c4_data_loader.train_loader):
            inputs, value_labels, policy_labels = data
            assert isinstance(inputs, Tensor)
            inputs = inputs.to('cuda')
            value_labels = value_labels.to('cuda')
            policy_labels = policy_labels.to('cuda')

            net.train()
            optimizer.zero_grad()
            policy_outputs, value_outputs = net(inputs)

            policy_loss = policy_criterion(policy_outputs, policy_labels)
            value_loss = value_criterion(value_outputs, value_labels)
            loss = policy_loss + value_loss * value_loss_lambda

            n = len(inputs)
            train_policy_loss_num += float(policy_loss.item()) * n
            train_value_loss_num += float(value_loss.item()) * n
            train_den += n
            train_policy_accuracy_num += get_num_correct_policy_predictions(policy_outputs, policy_labels)
            train_value_accuracy_num += get_num_correct_value_predictions(value_outputs, value_labels)

            loss.backward()
            optimizer.step()

        scheduler.step()
        train_policy_accuracy = train_policy_accuracy_num / train_den
        avg_train_policy_loss = train_policy_loss_num / train_den
        train_value_accuracy = train_value_accuracy_num / train_den
        avg_train_value_loss = train_value_loss_num / train_den

        with torch.set_grad_enabled(False):
            for data in c4_data_loader.test_loader:
                inputs, value_labels, policy_labels = data
                inputs = inputs.to('cuda')
                value_labels = value_labels.to('cuda')
                policy_labels = policy_labels.to('cuda')

                net.eval()
                policy_outputs, value_outputs = net(inputs)

                policy_loss = policy_criterion(policy_outputs, policy_labels)
                value_loss = value_criterion(value_outputs, value_labels)

                avg_test_policy_loss = policy_loss.item()
                avg_test_value_loss = value_loss.item()

                n = len(inputs)
                test_policy_accuracy = get_num_correct_policy_predictions(policy_outputs, policy_labels) / n
                test_value_accuracy = get_num_correct_value_predictions(value_outputs, value_labels) / n

                best = test_policy_accuracy > best_test_policy_accuracy
                print('%5d %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %s' %
                      (epoch,
                       avg_train_value_loss, train_value_accuracy,
                       avg_train_policy_loss, train_policy_accuracy,
                       avg_test_value_loss, test_value_accuracy,
                       avg_test_policy_loss, test_policy_accuracy,
                       '*' if best else ''
                       ))

                if best:
                    best_net = copy.deepcopy(net)
                    best_test_policy_accuracy = test_policy_accuracy

    t2 = time.time()
    sec_per_epoch = (t2 - t1) / num_epochs
    ms_per_train_row = 1000 * sec_per_epoch / c4_data_loader.train_n

    print('Finished Training (%.3fms/train-row)' % ms_per_train_row)
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
