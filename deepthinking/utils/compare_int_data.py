import math
import torch
from torch.utils import data

import torch.nn.functional as F

import os

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611

class CompareIntDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, prefix: str) -> None:
        # with 4 bits
        # 8! * (16 choose 8) inputs = a lot
        # (16 choose 8) solutions = 12870
        base_folder = "compare_int_data"
        inputs_path = os.path.join(root, base_folder, prefix + "_inputs.pkl")
        self.inputs = torch.load(inputs_path)

        targets_path = os.path.join(root, base_folder, prefix + "_targets.pkl")
        self.targets = torch.load(targets_path)

        self.inputs = self.inputs.float()
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        return True


def prepare_compare_int_loader(train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, shuffle=True):

    # train_1000000_examples_16_items_8_bits
    trainset = CompareIntDataset("../../../data", "train_1000000_examples_16_items_8_bits")
    valset = CompareIntDataset("../../../data", "val_10000_examples_16_items_8_bits")
    testset_big = CompareIntDataset("../../../data", "test_big_10000_examples_64_items_8_bits")
    testset_long = CompareIntDataset("../../../data", "test_long_10000_examples_16_items_16_bits")

    #train_split = int(train_split * len(dataset))

    """trainset, valset = torch.utils.data.random_split(dataset,
                                                     [train_split,
                                                      int(len(dataset) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))"""

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    valloader = data.DataLoader(valset, num_workers=0, batch_size=train_batch_size,
                                shuffle=False, drop_last=False)
    test_big_loader = data.DataLoader(testset_big, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    test_long_loader = data.DataLoader(testset_long, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    loaders = {
        "train": trainloader,
        "test_long": test_long_loader,
        "val": valloader,
        "test_big": test_big_loader
    }

    return loaders