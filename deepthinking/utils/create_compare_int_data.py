import torch
import math

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

def create_data(prefix, num_items: int = 8, num_examples = 10000, num_bits = 8):
    inputs = torch.zeros((num_examples, 1, num_items, num_bits), dtype=torch.long)
    targets = torch.zeros((num_examples, 1, num_items, num_bits), dtype=torch.long)
    
    MAX_NUM = math.pow(2, num_bits) - 1
    ones = torch.ones((int(MAX_NUM),))
    for i in range(num_examples):
        if i % 10000 == 0:
            print(i, "/", num_examples)
        nums = torch.multinomial(ones, num_items)
        sorted_nums, _ = torch.sort(nums)

        for j, num in enumerate(nums):
            inputs[i, 0, j] = torch.tensor(convert_int_to_bits(num, num_bits))

        for j, num in enumerate(sorted_nums):
            targets[i, 0, j] = torch.tensor(convert_int_to_bits(num, num_bits))
    torch.save(inputs.float(), prefix + "_" + str(num_examples) + "_examples_" + str(num_items) + "_items_" + str(num_bits) + "_bits_inputs.pt")
    torch.save(targets, prefix + "_" + str(num_examples) + "_examples_" + str(num_items) + "_items_" + str(num_bits) + "_bits_targets.pt")

create_data("train", num_items=16, num_examples=10000)
create_data("val", num_items=16, num_examples=10000)
create_data("test_big", num_items=64, num_examples=10000)
create_data("test_long", num_items=16, num_examples=10000, num_bits=16)
