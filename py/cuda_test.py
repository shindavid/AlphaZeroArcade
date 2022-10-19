import time

from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

for batch_size in (1, 64):
    for use_cuda in (False, True):
        model = models.resnet50()

        dataset = datasets.FakeData(
            size=1000,
            transform=transforms.ToTensor())
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True
        )

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

