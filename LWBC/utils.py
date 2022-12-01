import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_classifier(num_linear, f_dim, output_dim, linear_bias):
    if num_linear =='3linear':
        # print("3 Linear")
        classifier = torch.nn.Sequential(
            torch.nn.Linear(f_dim, int(f_dim/2), bias=linear_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(int(f_dim/2), int(f_dim/4), bias=linear_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(int(f_dim/4), output_dim, bias=linear_bias),
        )    
    elif num_linear =='2linear':
        # print("2 Linear")
        classifier = torch.nn.Sequential(
            torch.nn.Linear(f_dim, int(f_dim/2), bias=linear_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(int(f_dim/2), output_dim, bias=linear_bias),
        )    
    else:
        classifier = torch.nn.Sequential(
                torch.nn.Linear(f_dim, output_dim, bias=linear_bias)
            )
    return classifier

def ce(outputs, labels, reduce=True):
    return F.cross_entropy(outputs, labels, reduce=reduce)
   
#celebA
def num_correct_all(outputs,labels,sensitive, num_class, flag):
    if flag ==0:
        return torch.Tensor([0,0,0,0]), torch.Tensor([0,0,0,0])
    else:
        _, preds = torch.max(outputs, dim=1)
        idx1 = labels.cpu()==0 # non-blond_hair
        idx2 = labels.cpu()==1 # Blond_Hair
        b_idx1 = sensitive.cpu()==0 # Female
        b_idx2 = sensitive.cpu()==1 # Male
        correct = torch.Tensor([preds[(idx1*b_idx1)].eq(labels[(idx1*b_idx1)]).sum().cpu(), # non-blond & female
                    preds[(idx1*b_idx2)].eq(labels[(idx1*b_idx2)]).sum().cpu(), # non-blond & male
                    preds[(idx2*b_idx1)].eq(labels[(idx2*b_idx1)]).sum().cpu(), # blond & female
                    preds[(idx2*b_idx2)].eq(labels[(idx2*b_idx2)]).sum().cpu()]) # blond & male
        count = torch.Tensor([(idx1*b_idx1).sum(), 
                (idx1*b_idx2).sum(), 
                (idx2*b_idx1).sum(), 
                (idx2*b_idx2).sum()])
        return correct, count

#imagenet
def num_correct_cls(outputs,labels, sensitive, num_class, flag):
    if flag ==0:
        return torch.Tensor([0]*num_class), torch.Tensor([0]*num_class)
    else:
        _, preds = torch.max(outputs, dim=1)
        correct = []
        count = []
        for i in range(num_class):
            idx = labels.cpu()==i
            correct.append(preds[idx].eq(labels[idx]).sum().cpu())
            count.append(idx.sum())
        correct = torch.Tensor(correct)
        count = torch.Tensor(count)
        return correct, count

#imagenet
def imagenet_num_correct_all(outputs,labels,sensitive):
    _, preds = torch.max(outputs, dim=1)
    corrects = np.zeros([3, 9, 9])
    counts = np.zeros([3, 9, 9])
    for i in range(preds.shape[0]):
        c = preds[i].eq(labels[i]).cpu()
        corrects[0][labels[i],sensitive[0][i]] += c
        corrects[1][labels[i],sensitive[1][i]] += c
        corrects[2][labels[i],sensitive[2][i]] += c

        counts[0][labels[i],sensitive[0][i]] += 1
        counts[1][labels[i],sensitive[1][i]] += 1
        counts[2][labels[i],sensitive[2][i]] += 1

    return corrects, counts

def num_correct(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    correct = preds.eq(labels).sum()
    return correct

def get_class_num(dataset_name):
    if dataset_name =='celebA_Blond_Hair' or dataset_name =='celebA_Heavy_Makeup':
        output_dim = 2
    elif dataset_name =='BAR':
        output_dim = 6
    elif dataset_name =='NICO':
        output_dim = 10
    elif dataset_name =='imagenet':
        output_dim = 9
    elif dataset_name =='waterbirds':
        output_dim = 2
    return output_dim
