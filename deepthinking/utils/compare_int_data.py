import math
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
        num = num << 1
        num += int(b)
    return num

def convert_int_to_bits(num, num_bits = 8):
    bits = []
    curr_num = num
    for _ in range(num_bits):
        bits.append(curr_num % 2)
        curr_num = curr_num >> 1
    bits.reverse()
    assert convert_bits_to_int(bits) == num
    return bits

class CompareIntDataset(torch.utils.data.Dataset):
    base_folder = "compare_int_data"

    def __init__(self, root: str, num_items: int = 8, num_examples = 10000, num_bits = 8) -> None:
        # with 4 bits
        # 8! * (16 choose 8) inputs = a lot
        # (16 choose 8) solutions = 12870
        self.inputs = torch.zeros((num_examples, 1, num_items, num_bits), dtype=torch.long)
        self.targets = torch.zeros((num_examples, 1, num_items, num_bits), dtype=torch.long)
        
        MAX_NUM = math.pow(2, num_bits) - 1
        ones = torch.ones((int(MAX_NUM),))
        for i in range(num_examples):
            if i % 10000 == 0:
                print(i, "/", num_examples)
            nums = torch.multinomial(ones, num_items)
            sorted_nums, _ = torch.sort(nums)

            for j, num in enumerate(nums):
                self.inputs[i, 0, j] = torch.tensor(convert_int_to_bits(num, num_bits))

            for j, num in enumerate(sorted_nums):
                self.targets[i, 0, j] = torch.tensor(convert_int_to_bits(num, num_bits))
        

        self.inputs = self.inputs.float()
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        return True


def prepare_compare_int_loader(train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, train_and_val_examples=500, test_examples=500, shuffle=True):

    dataset = CompareIntDataset("../../../data", num_items=train_data, num_examples=train_and_val_examples)
    testset = CompareIntDataset("../../../data", num_items=test_data, num_examples=test_examples)
    testset_16 = CompareIntDataset("../../../data", num_items=train_data, num_examples=test_examples, num_bits=16)

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
    test16loader = data.DataLoader(testset_16, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    loaders = {"train": trainloader, "test": testloader, "val": valloader, "test_16": test16loader}

    return loaders