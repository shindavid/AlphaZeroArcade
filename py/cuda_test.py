#!/usr/bin/env python3
import time

import torch
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

USE_LOADER = True
torch.set_grad_enabled(False)

for use_cuda in (True, False):
    for batch_size in (1, 64):
        model = models.resnet50()
        model.eval()
        size = 1024

        if USE_LOADER:
            dataset = datasets.FakeData(
                size=size,
                transform=transforms.ToTensor())
            loader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True
            )
        else:
            loader = [(torch.rand(batch_size, 3, 224, 224), None) for _ in range(size // batch_size)]

        if use_cuda:
            model.to('cuda')

        x = 0
        t1 = time.time()
        if use_cuda:
            for data, _ in loader:
                data = data.to('cuda', non_blocking=True)
                x += torch.sum(model(data).to('cpu'))
        else:
            for data, _ in loader:
                x += torch.sum(model(data))

        t2 = time.time()
        t = t2 - t1
        print('batch_size:%-2d use_cuda:%d runtime:%.3fs' % (batch_size, use_cuda, t))

