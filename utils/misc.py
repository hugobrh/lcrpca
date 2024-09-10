#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:57:48 2023

@author: hugobrehier
"""

import torch
import torch.nn as nn

import itertools,os

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s_(%d).%s" % (basename, next(c), ext)
    return actualname


def norm_sparse(X,mode='l1'):
    '''sparse norms over batches (BS,CH,H,W)'''
    if mode == 'l1':
        return X.squeeze().abs().sum()
    if mode == 'l21':
        return X.squeeze().norm(dim=1).sum()

def norm_nuc(X):
    '''nuclear norm over batches (BS,CH,H,W)'''
    assert X.shape[1] == 1, 'norm nuc only works on single channel batches'
    return torch.linalg.norm(X.squeeze(),ord='nuc',dim=(1,2)).sum()


class DiceLoss(nn.Module):
    ''' Loss based on Dice coefficient '''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection)/(inputs.square().sum() + targets.square().sum())  
        
        return 1-dice
    
    
def TCR(Xtrue,X):
    '''Target to Clutter Ratio '''
    if type(X) is not torch.Tensor:
        X = torch.tensor(X).squeeze()
    if type(Xtrue) is not torch.Tensor:
        Xtrue = torch.tensor(Xtrue).squeeze()

    Xtrue = Xtrue.int().bool()
    Xtarget  = X[Xtrue].square().mean()
    Xclutter = X[~Xtrue].square().mean()
    
    return 10*torch.log10(Xtarget/Xclutter)

