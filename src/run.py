# !pip -q install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
# !pip -q install geffnet
# !pip install -U git+https://github.com/albu/albumentations --no-cache-dir
from conf import *
from loader import *
from models import *
from trainer import *
from loss import *
from utils import *
from scheduler import *

import random

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image

from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

import albumentations as A
import geffnet

from sklearn.model_selection import StratifiedKFold

def main():
    set_seed(args.seed)

    train = pd.read_csv('../data/public/train.csv')
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    train['fold'] = 0
    for idx, [trn, val] in enumerate(skf.split(train, train['landmark_id'])):
        train.loc[val, 'fold'] = idx
    train['filepath'] = [os.path.join('../data/train', str(lm_id), str(id)+'.JPG') for lm_id, id in zip(train['landmark_id'], train['id'])]

    if args.class_weights == "log":
        val_counts = train.landmark_id.value_counts().sort_index().values
        class_weights = 1/np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None
    
    # sub = pd.read_csv('../submit/baseline_9.csv')
    # sub['filepath'] = [os.path.join('../data/public/test', id, folder+'.JPG') for id, folder in zip(sub['id'].apply(lambda x: x[0]), sub['id'])]
    # sub.loc[37931, 'landmark_id']=901
    # sub.loc[299, 'landmark_id']=275
    # sub.loc[301, 'landmark_id']=275
    # sub.loc[605, 'landmark_id']=120
    # sub.loc[939, 'landmark_id']=666
    # sub.loc[956, 'landmark_id']=671
    # sub.loc[1705:1730, 'landmark_id']=751
    # sub.loc[1776, 'landmark_id']=909
    # sub.loc[1825, 'landmark_id']=159
    # sub.loc[2065, 'landmark_id']=348
    # sub.loc[1745, 'landmark_id']=513
    # sub.loc[3154:3163, 'landmark_id']=487
    # sub.loc[3213:3221, 'landmark_id']=470
    # sub.loc[3811:3847, 'landmark_id']=1046
    # sub.loc[1825, 'landmark_id']=606
    # sub.loc[2487, 'landmark_id']=646
    # sub.loc[2892:2927, 'landmark_id']=988
    # sub.loc[1902, 'landmark_id']=579
    # sub.loc[12223:12262, 'landmark_id']=543
    # sub.loc[2744, 'landmark_id']=812
    # sub.loc[2488, 'landmark_id']=646
    # sub.loc[3202, 'landmark_id']=470
    # sub.loc[3008, 'landmark_id']=292
    # sub.loc[2155:2164, 'landmark_id']=617
    # sub.loc[4259, 'landmark_id']=578
    # sub.loc[3252, 'landmark_id']=219
    # sub.loc[3807, 'landmark_id']=838
    # sub.loc[4368:4383, 'landmark_id']=868
    # sub.loc[4777, 'landmark_id']=64
    # sub.loc[5382:5401, 'landmark_id']=312
    # sub.loc[3510, 'landmark_id']=826
    # sub.loc[4086:4094, 'landmark_id']=398
    # sub.loc[5204, 'landmark_id']=589
    # sub.loc[5836, 'landmark_id']=1023
    # sub.loc[3289, 'landmark_id']=109
    # sub.loc[34109, 'landmark_id']=2
    # sub.loc[6628, 'landmark_id']=795
    # sub.loc[4866, 'landmark_id']=176
    # sub.loc[6882, 'landmark_id']=27
    # sub.loc[16638:16671, 'landmark_id']=40
    # sub.loc[16009:16043, 'landmark_id']=892
    # sub.loc[16048:16076, 'landmark_id']=861
    # sub.loc[21474, 'landmark_id']=66
    # sub.loc[21489:21528, 'landmark_id']=502
    # sub.loc[36193:36212, 'landmark_id']=236
    # sub.loc[19344:19358, 'landmark_id']=488
    # sub.loc[11882, 'landmark_id']=1008
    # sub.loc[36980:36987, 'landmark_id']=29
    # sub.loc[24477, 'landmark_id']=10
    # sub.loc[16323:16338, 'landmark_id']=136
    # sub.loc[7392, 'landmark_id']=437
    # sub.loc[14800:14806, 'landmark_id']=616
    # sub.loc[19331, 'landmark_id']=488
    # sub.loc[37440:37474, 'landmark_id']=38
    # sub.loc[19373, 'landmark_id']=757
    # sub.loc[15640:15675, 'landmark_id']=371
    # sub.loc[25803:25833, 'landmark_id']=771
    # sub.loc[30633:30678, 'landmark_id']=763
    # sub.loc[17764, 'landmark_id']=539
    # sub.loc[17172, 'landmark_id']=105
    # sub.loc[19420, 'landmark_id']=258
    # sub.loc[20568, 'landmark_id']=710
    # sub.loc[13187, 'landmark_id']=538
    # sub.loc[28293:28296, 'landmark_id']=90
    # sub.loc[37072, 'landmark_id']=607
    # sub.loc[35413:35435, 'landmark_id']=595
    # sub.loc[34776, 'landmark_id']=297
    # sub.loc[33846:33848, 'landmark_id']=32
    # sub.loc[37556, 'landmark_id']=998
    # sub.loc[35557, 'landmark_id']=982
    # sub.loc[33040, 'landmark_id']=867
    # sub.loc[33174, 'landmark_id']=107
    # sub.loc[30500, 'landmark_id']=246
    # sub.loc[27948, 'landmark_id']=328
    # sub.loc[33198, 'landmark_id']=107
    # sub.loc[31883, 'landmark_id']=153
    # sub.loc[31885, 'landmark_id']=153
    # sub.loc[26452, 'landmark_id']=447
    # sub.loc[25295, 'landmark_id']=253
    # sub.loc[1879:1880, 'landmark_id']=579
    # drop_idx = [ 2465,  2928,  3012,  3154,  3227,  3467,  3811,  3966,  4083, 6549,  6627,  6647,  6648,  6649,  6848,  7615,  8276,  8383, 8389,  8390,  9254, 10412, 10506, 12295, 12302, 12303, 12393, 12415, 12416, 12419, 12420, 12707, 12893, 13516, 13539, 13540, 14783, 14820, 16638, 17754, 17797, 17798, 19256, 19396, 19407,
    #         19408, 20360, 21053, 21488, 25288, 25302, 25303, 25871, 26154,27780, 27801, 27802, 27803, 27820, 28217, 28248, 28249, 28387,28388, 30633, 33507, 33634, 34952, 36193, 37617]
    # sub = sub.drop(drop_idx).reset_index(drop=True)

    # sub = pd.read_csv('../submit/baseline_9.csv')
    # sub['filepath'] = [os.path.join('../data/public/test', id, folder+'.JPG') for id, folder in zip(sub['id'].apply(lambda x: x[0]), sub['id'])]
    # train = pd.read_csv('../data/public/new_train.csv')
    # train['filepath'] = [os.path.join('../data/train', str(lm_id), str(id)+'.JPG') for lm_id, id in zip(train['landmark_id'], train['id'])]
    new_train = pd.read_csv('../data/public/new_train.csv')


    trn = train.loc[train['fold']!=args.fold].reset_index(drop=True)
    trn = trn.loc[~np.isin(trn.index, new_train.index)].reset_index(drop=True)
    # trn = pd.concat([trn, sub[sub['conf']>0.15]]).reset_index(drop=True)
    val = train.loc[train['fold']==args.fold].reset_index(drop=True)

    # trn = train
    # val = sub

    print(f'trn size : {trn.landmark_id.nunique()}, last batch size : {trn.shape[0]%args.batch_size}') #: 1049
    # print(len(trn)) #: 70481
    # image size : (540, 960, 3)
    
    if args.DEBUG:
        trn = trn.iloc[:2500]
        val = val.iloc[:2500]
    
    train_dataset = LMDataset(trn, aug=args.tr_aug, normalization=args.normalization)
    valid_dataset = LMDataset(val, aug=args.val_aug, normalization=args.normalization)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False)

    model = Net(args)
    model = model.to(args.device)
    # model.load_state_dict(torch.load('/home/hhl/바탕화면/dacon/dacon21/model/new_train/best_tf_efficientnet_b1_ns_best_fold_0.pth'))

    # optimizer definition
    metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
    metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=None, reduction="sum")
    if args.optim=='sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.optim=='adamw':
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_epo)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
    
    optimizer.zero_grad()
    optimizer.step()
    
    val_pp = 0.
    model_file = f'../model/{args.backbone}_best_fold_{args.fold}.pth'
    for epoch in range(1, args.cosine_epo+args.warmup_epo+1+49):
        # print(optimizer.param_groups[0]['lr'])

        # scheduler_cosine.step(epoch-1)
        scheduler_warmup.step(epoch-1)
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_epoch(metric_crit, epoch, model, train_loader, optimizer)
        if epoch>1:
            val_outputs = val_epoch(metric_crit_val, model, valid_loader)
            np.save('../submit/val_outputs_best.npy', val_outputs)
            results = val_end(val_outputs)
            print(results)

            val_loss = results['val_loss']
            val_gap = results['val_gap']
            # val_gap_landmarks = results['val_gap_landmarks']
            # val_gap_pp = results['val_gap_pp']
            # val_gap_landmarks_pp = results['val_gap_landmarks_pp']
            # np.save('../submit/val_outputs.npy', val_outputs)
            # content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}, val_gap: {val_gap:.4f}, val_gap_landmarks: {val_gap_landmarks:.4f}, val_gap_pp: {val_gap_pp:.4f}, val_gap_landmarks_pp: {val_gap_landmarks_pp:.4f}'
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}, val_gap: {val_gap:.4f}'
            print(content)
            with open(f'../model/log_fold_{args.backbone}_{args.fold}.txt', 'a') as appender:
                appender.write(content + '\n')
            
            val_gap_pp = val_gap
            if val_gap_pp > val_pp:
                print('val_gap_pp_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_pp, val_gap_pp))
                torch.save(model.state_dict(), model_file)
                val_pp = val_gap_pp

        
    
if __name__ == '__main__':
    main()