import torch
from torch.utils import data

import torch.nn.functional as F

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611

def convert_bits_to_int(bits):
    num = 0
    for b in bits:
        num << 1
        num += int(b)
    return num

class CompareIntDataset(torch.utils.data.Dataset):
    base_folder = "compare_int_data"

    def __init__(self, root: str, num_bits: int = 32) -> None:
        num_examples = 10000
        self.inputs = torch.randint(0, 2, (num_examples, 2, num_bits), dtype=torch.long)

        targets = []
        for i in range(num_examples):
            a = self.inputs[i][0].clone()
            b = self.inputs[i][1].clone()
            if convert_bits_to_int(a) < convert_bits_to_int(b):
                targets.append(torch.stack([a, b]))
            else:
                targets.append(torch.stack([b, a]))
        self.targets = targets
        self.inputs = self.inputs.float()
        print("sszzz", self.targets[0].size())
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        return True


def prepare_compare_int_loader(train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, shuffle=True):

    dataset = CompareIntDataset("../../../data", num_bits=train_data)
    testset = CompareIntDataset("../../../data", num_bits=test_data)

    train_split = int(train_split * len(dataset))

    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [train_split,
                                                      int(len(dataset) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    valloader = data.DataLoader(valset, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders