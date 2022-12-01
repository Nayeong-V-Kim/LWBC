from bdb import set_trace
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import wandb

from collections import defaultdict
from tqdm import tqdm
from torch.utils import data

import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
from configs import *
import warnings
warnings.filterwarnings("ignore")


class dataread():
    def __init__(self, dataset, mode, num_classifier = 5, set_size=40):
        if dataset=='celebA_Blond_Hair':           
            npy_dict = {}
            npyfile_dir = 'npy_files/celebA_Blond_Hair'
            attr = 'Blond_Hair'
        if dataset=='celebA_Heavy_Makeup':           
            npy_dict = {}
            npyfile_dir = 'npy_files/celebA_Heavy_Makeup'
            attr = 'Heavy_Makeup'
        elif dataset=='BAR':
            npy_dict = {}
            npyfile_dir = 'npy_files/BAR'
            attr = 'BAR'
        elif dataset=='NICO':
            npy_dict = {}
            npyfile_dir = 'npy_files/NICO'
            attr = 'NICO'
        elif dataset=='imagenet':
            npy_dict = {}
            npyfile_dir = 'npy_files/imagenet'
            attr = 'imagenet'

        npy_name_list = os.listdir(npyfile_dir)
        for npy_name in npy_name_list:
            if npy_name.find(attr) != -1:
                dtype1 = None; dtype2= None
                for d1 in ["train", "valid", "test"]:    
                    if npy_name.find(d1) != -1:
                        dtype1 = d1
                        for d2 in ["feature","target" ,"bias", "path"]:
                            if npy_name.find(d2) != -1:
                                dtype2 = d2
                                npy_dict["{}_{}".format(dtype1, dtype2)] = npy_name
                                break
                        break
        
        self.feature = np.load(os.path.join(npyfile_dir, npy_dict["{}_feature".format(mode)]))
        self.target = np.load(os.path.join(npyfile_dir, npy_dict["{}_target".format(mode)]))
        self.bias = np.load(os.path.join(npyfile_dir, npy_dict["{}_bias".format(mode)]))
        self.path = np.load(os.path.join(npyfile_dir, npy_dict["{}_path".format(mode)]))

        masks = np.array([[]]*num_classifier).tolist()
        nlbl = len(set(self.target))
        for i in range(num_classifier):
            masks[i].append(np.random.choice(np.arange(len(self.feature)), set_size*nlbl))

        self.masks = np.stack(masks).reshape(num_classifier,-1)
        self.masks_place = torch.zeros(num_classifier, len(self.feature)).bool()
        for i in range(num_classifier):
            self.masks_place[i, self.masks[i]] = True   
        self.num_classifier = num_classifier
    
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        img = self.feature[index]
        target = self.target[index]
        context = self.bias[index]
        mask = self.masks_place[:, index].tolist()
        return img, target, context, mask
