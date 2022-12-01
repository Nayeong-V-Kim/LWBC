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
from dataloader import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(3, suppress=True)
device = torch.device('cuda')

seed = config.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

if not config.local:
    wandb.init(
        config=config,
        project='LWBC_BAR',
        anonymous='allow',
    )

config.dataset = 'BAR'
val_metric = 'val_acc'
test_metric = 'test_acc'

train_dataset = dataread(dataset=config.dataset, mode='train', num_classifier=config.num_classifier, set_size=config.set_size)
train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
save_train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
valid_dataset = dataread(dataset=config.dataset, mode='valid')
valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
test_dataset = dataread(dataset=config.dataset, mode='test')
test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

f_dim = 512
output_dim = get_class_num(config.dataset)
num_classifier = config.num_classifier
classifiers = []

for _ in range(num_classifier):
    classifiers.append(get_classifier(config.classifier, f_dim, output_dim, config.linear_bias).to(device))
target_classifier = get_classifier(config.classifier, f_dim, output_dim, config.linear_bias).to(device)

params = []
for i in range(num_classifier):
    params += list(classifiers[i].parameters())

if config.optimizer =='SGD':
    print("SGD")
    optimizer=torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.SGD(target_classifier.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
else:
    optimizer=torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.Adam(target_classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)

kl_loss = torch.nn.KLDivLoss(reduce=False)

best_metric = {}
best_metric[val_metric] = 0 

for epoch in range(config.epochs):
    losses = AverageMeter()
    target_losses = AverageMeter()
    kd_losses = AverageMeter()
    ens_losses = AverageMeter()
    for i in range(num_classifier):
        classifiers[i].train()
    target_classifier.train()
    weights = []
    num_samples = 0
    tot_correct = torch.Tensor([0]*(num_classifier+1))
    for data, labels, sensitive, mask in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        optimizer_target.zero_grad()
        data = data.to(device)
        labels = labels.to(device)
        loss = 0
        num_samples += data.shape[0]
        outputs =[]
        loss_cand = []
        for i in range(num_classifier):
            outputs.append(classifiers[i](data)*config.temperature)    
            tot_correct[i] += num_correct(outputs[i], labels).item()
            count = mask[i].sum()
            if count>0:
                loss += (ce(outputs[i], labels, False)*mask[i].to(device)).sum()/mask[i].sum()

        if epoch+1 > config.warmup:
            target_outputs = target_classifier(data)*config.temperature
            tot_correct[-1] += num_correct(target_outputs, labels).item()
            weight = (1/(torch.stack([outputs[i].argmax(dim=1).eq(labels) for i in range(num_classifier)]).float().sum(dim=0)/num_classifier+config.alpha))
            weights.extend(weight.detach().cpu().numpy().tolist())
            target_loss = (ce(target_outputs, labels, False)*weight).mean()*config.loss_scale
            
            target_losses.update(target_loss.item())
            target_loss.backward()
            optimizer_target.step()

            target_outputs = target_classifier(data)*config.temperature
            if config.kd_lambda > 0 and epoch+1 > config.warmup+1:
                kl_outputs = F.softmax(target_outputs.detach() / config.kd_temperature)
                loss = loss * (1-config.kd_lambda)
                ens_losses.update(loss)
                for i in range(num_classifier):
                    kd_loss_ = kl_loss(F.log_softmax(outputs[i] / config.kd_temperature), kl_outputs) * (~mask[i][..., None].to(device))
                    kd_loss = config.kd_lambda * kd_loss_.mean() * (config.kd_temperature ** 2) 
                    
                    loss += kd_loss
                    kd_losses.update(kd_loss.item())
                    
            else:
                ens_losses.update(loss)
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        mean_weight = np.mean(weights)

    train_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
          
    metrics = {
        'train_loss': losses.avg,
        'train_target_loss': target_losses.avg,
        'weights': mean_weight,
        'kd_loss': kd_losses.avg*num_classifier,
        'ens_loss': ens_losses.avg/num_classifier
    }

    # Eval
    # if epoch+1 > config.warmup:
    if epoch+1 > -1:
        num_samples = 0
        tot_correct = torch.Tensor([0]*(num_classifier+1))
        for data, labels, sensitive,_ in tqdm(valid_loader, leave=False):
            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.shape[0]
            num_samples += batch_size
            for k, classifier in enumerate(classifiers):
                classifier.eval()
                with torch.no_grad():
                    outputs = classifier(data)  
                
                tot_correct[k] += num_correct(outputs, labels).item()
            target_classifier.eval()
            with torch.no_grad():
                outputs = target_classifier(data)  
            
            tot_correct[-1] += num_correct(outputs, labels).item()
        valid_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
        
        num_samples = 0
        tot_correct = torch.Tensor([0]*(num_classifier+1))
        for data, labels, sensitive,_ in tqdm(test_loader, leave=False):
            data = data.to(device)
            labels = labels.to(device)
            batch_size = data.shape[0]
            num_samples += batch_size
            for k, classifier in enumerate(classifiers):
                classifier.eval()
                with torch.no_grad():
                    outputs = classifier(data)  
                
                tot_correct[k] += num_correct(outputs, labels).item()
            target_classifier.eval()
            with torch.no_grad():
                outputs = target_classifier(data)  
            
            tot_correct[-1] += num_correct(outputs, labels).item()
        avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
        
        if config.save:
            np.save('{}/checkpoints/{}_{}.pth'.format(config.save_path, config.dataset, epoch), target_classifier.state_dict())
            for cls_number in range(num_classifier):
                np.save('{}/checkpoints/{}_{}.pth'.format(config.save_path, config.dataset, epoch), classifiers[cls_number].state_dict())
        print("Epoch {} Train loss: {:.4f} target_loss: {:.4f} train_acc {:4f} val_acc {:4f}".format(epoch+1, losses.avg, target_losses.avg, train_avg_accuracy[-1], valid_avg_accuracy[-1]))
        
        metrics['train_acc'] = train_avg_accuracy[-1]
        metrics['val_acc'] = valid_avg_accuracy[-1]
        
        for k in range(num_classifier):
            metrics['train_acc_M{}'.format(k)]=train_avg_accuracy[k]
            metrics['val_acc_M{}'.format(k)]=valid_avg_accuracy[k]
        
        if best_metric[val_metric] <= metrics[val_metric]:
            best_metric = metrics.copy()
            best_metric['epoch'] = epoch+1
            best_metric[test_metric] = avg_accuracy[-1]
        
    else:
        print("Epoch {} Train_loss: ".format(epoch+1), losses.avg)

    if not config.local:
        wandb.log(metrics)

print("FINISH")
print("best epoch {} {} {:.4f} {} {:.4f}".format(best_metric['epoch'], val_metric, best_metric[val_metric], test_metric, best_metric[test_metric].item()))
