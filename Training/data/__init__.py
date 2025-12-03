import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .custom_transforms import *
from .datasets import RealFakeDataset, custom_collate_fn


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt, preprocess=None, return_dataset=False):

    shuffle = True if opt.isTrain else False

    batch_size = opt.batch_size // 6
    dataset = RealFakeDataset(opt)

    sampler = None
    print(len(dataset))
    if return_dataset:
        return dataset

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=opt.isTrain,
        collate_fn=custom_collate_fn,
    )
    return data_loader
