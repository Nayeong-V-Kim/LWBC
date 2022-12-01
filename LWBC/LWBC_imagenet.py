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
        project='LWBC_imagenet',
        anonymous='allow',
    )


EVALSPLIT = num_correct_cls
split_num = 9
val_metric = 'val_acc'
test_imagenet9_metric = 'val_unbiased_acc'
test_imagenetA_metric = 'test_acc'
    
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
    mask_sum = torch.Tensor([0]*num_classifier)
    weights = []
    num_samples = 0
    tot_correct = torch.Tensor([0]*(num_classifier+1))
    correct_split = torch.zeros(num_classifier+1, split_num)
    count_split = torch.zeros(num_classifier+1, split_num)
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
            s_correct, s_count =  EVALSPLIT(outputs[i],labels, sensitive, output_dim, 1)
            correct_split[i] += s_correct
            count_split[i] += s_count 
            count = mask[i].sum()
            if count>0:
                loss += (ce(outputs[i], labels, False)*mask[i].to(device)).sum()/mask[i].sum()
        
        if epoch+1 > config.warmup:
            target_outputs = target_classifier(data)*config.temperature
            tot_correct[-1] += num_correct(target_outputs, labels).item()
            s_correct, s_count =  EVALSPLIT(target_outputs,labels,sensitive, output_dim, 1)
            correct_split[-1] += s_correct
            count_split[-1] += s_count
            weight = (1/(torch.stack([outputs[i].argmax(dim=1).eq(labels) for i in range(num_classifier)]).float().sum(dim=0)/num_classifier+config.alpha))
            weights.extend(weight.detach().cpu().numpy().tolist())
            target_loss = (ce(target_outputs, labels, False)*weight).mean()*config.loss_scale
            
            target_losses.update(target_loss.item())
            target_loss.backward()
            optimizer_target.step()
            
            target_outputs = target_classifier(data)*config.temperature
            if config.kd_lambda >0 and epoch+1 > config.warmup+1:
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
    train_avg_accuracy_split = [correct_split[k] / count_split[k] for k in range(num_classifier+1)]
            
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
        corrects = np.zeros([num_classifier+1, 3, 9, 9])
        counts = np.zeros([num_classifier+1, 3, 9, 9])
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
                s_correct, s_count =  imagenet_num_correct_all(outputs,labels, sensitive.T)
                corrects[k] += s_correct
                counts[k] += s_count
            target_classifier.eval()
            with torch.no_grad():
                outputs = target_classifier(data)  
            tot_correct[-1] += num_correct(outputs, labels).item()
            s_correct, s_count =  imagenet_num_correct_all(outputs,labels, sensitive.T)
            corrects[-1] += s_correct
            counts[-1] += s_count
        valid_avg_accuracy =[]  
        valid_unbiased_acc =[] 
        for k in range(num_classifier+1):
            idx0 = counts[k, 0]>=10
            idx1 = counts[k, 1]>=10
            idx2 = counts[k, 2]>=10
            valid_unbiased_acc.append(((corrects[k, 0][idx0]/counts[k, 0][idx0]).mean()+(corrects[k, 1][idx1]/counts[k, 1][idx1]).mean()+(corrects[k, 2][idx2]/counts[k, 2][idx2]).mean())/3)
        valid_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
        

        num_samples = 0
        tot_correct = torch.Tensor([0]*(num_classifier+1))
        correct_split = torch.zeros(num_classifier+1, split_num)
        count_split = torch.zeros(num_classifier+1, split_num)
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
                s_correct, s_count =  EVALSPLIT(outputs,labels,sensitive, output_dim, 1)
                correct_split[k] += s_correct
                count_split[k] += s_count
            target_classifier.eval()
            with torch.no_grad():
                outputs = target_classifier(data)  
            
            tot_correct[-1] += num_correct(outputs, labels).item()
            s_correct, s_count =  EVALSPLIT(outputs,labels,sensitive, output_dim, 1)
            correct_split[-1] += s_correct
            count_split[-1] += s_count
        avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
        avg_accuracy_split = [correct_split[k] / count_split[k] for k in range(num_classifier+1)]
        
        if config.save:
            np.save('{}/LWBC/{}_{}.pth'.format(config.save_path, config.dataset, epoch), target_classifier.state_dict())
        
        print("Epoch {} mean_weight {:.2f} Train loss: {:.4f} target_loss: {:.4f} train_acc {:.4f} val_acc {:.4f}".format(epoch+1, mean_weight, losses.avg, target_losses.avg, train_avg_accuracy[-1], valid_avg_accuracy[-1]))

        
        metrics['train_acc'] = train_avg_accuracy[-1]
        metrics['val_acc'] = valid_avg_accuracy[-1]
        # metrics['val_unbiased_acc'] = valid_unbiased_acc[-1]
        # metrics['test_acc'] = avg_accuracy[-1]
        
        for k in range(num_classifier):
            metrics['train_acc_M{}'.format(k)]=train_avg_accuracy[k]
            metrics['val_acc_M{}'.format(k)]=valid_avg_accuracy[k]
            
        if best_metric[val_metric] < metrics[val_metric]:
            best_metric = metrics.copy()
            best_metric['epoch'] = epoch+1
            best_metric[test_imagenet9_metric] = valid_unbiased_acc[-1]
            best_metric[test_imagenetA_metric] = avg_accuracy[-1]
        print("best epoch {} {} {:.4f} {} {:.4f} {} {:.4f}".format(best_metric['epoch'], val_metric, best_metric[val_metric], test_imagenet9_metric, best_metric[test_imagenet9_metric].item(), test_imagenetA_metric, best_metric[test_imagenetA_metric].item()))
    else:
        print("Epoch {} Train_loss: ".format(epoch+1), losses.avg)

    if not config.local:
        wandb.log(metrics)

print("FINISH!")
print("best epoch {} {} {:.4f} {} {:.4f} {} {:.4f}".format(best_metric['epoch'], val_metric, best_metric[val_metric], test_imagenet9_metric, best_metric[test_imagenet9_metric].item(), test_imagenetA_metric, best_metric[test_imagenetA_metric].item()))








        # # if epoch % 10 ==0:
        # results = np.array([[]]*(num_classifier+1)).tolist()
        # corrects = np.array([[]]*(num_classifier+1)).tolist()
        # for data, labels, sensitive, _ in tqdm(save_train_loader, leave=False):
        #     data, labels = data.to(device), labels.to(device)
                
        #     for group_num in range(num_classifier):
        #         with torch.no_grad():
        #             outputs = classifiers[group_num](data)
        #             results[group_num].append(outputs.detach().cpu())
        #             corrects[group_num].append(outputs.argmax(dim=1).eq(labels).detach().cpu())
        #     outputs = target_classifier(data)
        #     results[-1].append(outputs.detach().cpu())
        #     corrects[-1].append(outputs.argmax(dim=1).eq(labels).detach().cpu())
                    
        # for group_num in range(num_classifier+1):
        #     results[group_num] = torch.cat(results[group_num])
        #     corrects[group_num] = torch.cat(corrects[group_num]).float().mean().item()

        # results = torch.stack(results).numpy()