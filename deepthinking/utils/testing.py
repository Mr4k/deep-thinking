""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from distutils.command.build_scripts import first_line_re
import einops
from matplotlib.pyplot import axis
import torch
from icecream import ic
from tqdm import tqdm
import wandb

import numpy as np

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(net, loaders, mode, iters, problem, device):
    accs = []
    for (loader, test_type) in loaders:
        if mode == "default":
            accuracy = test_default(net, loader, iters, problem, device, test_type)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted


def test_default(net, testloader, iters, problem, device, test_type):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    reporting_data = []

    first_batch = True
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs = net(inputs, iters_to_do=max_iters)

            batch_size = inputs.shape[0]
            num_data_pieces_to_report = min(10, batch_size) 
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()
            if first_batch:
                for j in range(num_data_pieces_to_report):
                    predicted_vid = []
                    for i in range(all_outputs.size(1)):
                        outputs = all_outputs[:, i]
                        predicted = get_predicted(inputs, outputs, problem)
                        in_shape = inputs[j].shape[0:]
                        sampled_input = inputs[j,0].int()
                        sampled_pred = predicted[j].view(-1, *in_shape)
                        sampled_target = targets[j].view(-1, *in_shape)
                        predicted_vid.append(sampled_pred.cpu().numpy())

                        percentage_correct_bits = (sampled_pred == sampled_target).sum() / reduce(operator.mul, in_shape, 1) * 100
                        
                    reporting_data.append((
                        wandb.Image(sampled_input.cpu().numpy()),
                        wandb.Image(sampled_pred.cpu().numpy()),
                        wandb.Video(np.stack(predicted_vid, axis=-1)),
                        wandb.Image(sampled_target.cpu().numpy()),
                        percentage_correct_bits,
                        test_type))
                first_batch = False
            total += targets.size(0)
    
    test_table = wandb.Table(data=reporting_data, columns=["inputs", "predicted", "animated pred", "labels", "percentage correct bits", "test_type"])
    wandb.log({"sample_outputs": test_table})

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc
