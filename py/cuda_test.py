#!/usr/bin/env python3
import time

import torch
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

USE_LOADER = False


for batch_size in (1, 64):
    for use_cuda in (False, True):
        model = models.resnet50()
        size = 1024

        if USE_LOADER:
            dataset = datasets.FakeData(
                size=1000,
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

        t1 = time.time()
        if use_cuda:
            for data, _ in loader:
                data = data.to('cuda', non_blocking=True)
                model(data)
        else:
            for data, _ in loader:
                model(data)

        t2 = time.time()
        t = t2 - t1
        print('batch_size:%-2d use_cuda:%d runtime:%.3fs' % (batch_size, use_cuda, t))

