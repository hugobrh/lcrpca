# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class ISTACell(nn.Module):
    def __init__(self,in_ch,out_ch, kc, ks,rk):
        super().__init__()
        
        #Complex input ok in nn.conv2D
        self.convd=nn.Conv2d(kc, out_ch, ks, stride=1, padding=ks//2)
        
        self.conve=nn.Conv2d(in_ch, kc, ks, stride=1, padding=ks//2)        
        self.conve.weight.data = torch.flip(torch.transpose(0.1*self.convd.weight.data,
                                                            -2,-1),[-2,-1]).transpose(0,1)
        self.coef0 = nn.Parameter(torch.tensor(0.1))
        self.thR   = nn.Parameter(torch.tensor(0.1))
        self.thL   = nn.Parameter(torch.tensor(0.1))
        
        self.relu = nn.ReLU()
        self.rk = rk
        
        self.features = None
        
    def forward(self,data):
        Y= data[0]
        L= data[1]
        R= data[2]
        
        L= self.svd_thres(L + self.coef0 * (Y - L - self.convd(R)),self.thL)
        R= self.sft_thres(R + self.conve(Y - L - self.convd(R)),self.thR)

        data[1]=L
        data[2]=R
        
        self.layer_features = R
        
        return data
                
    def svd_thres(self,Y,th):
        #svd grads have numerical instabilities cf warnings on torch.linalg.svd
        #use low rank pca instead which uses randomized svd
        #does work without centering (uses https://arxiv.org/pdf/0909.4061.pdf Algo 5.1)
        
        U,s,V = torch.pca_lowrank(Y.squeeze(),q=self.rk,center=False) 
        sm = s.max(1,keepdim=True).values.repeat(1,self.rk)
        s = torch.sgn(s) * self.relu(torch.abs(s) - th*sm)
        Y_th = U @ torch.diag_embed(s) @ V.mH
        return Y_th.reshape(Y.shape)
        
    def sft_thres(self,Y,th):
        BS,CH,H,W = Y.shape
        Ytmp = Y.reshape((BS,CH*H*W))
        Ym = Ytmp.max(1,keepdim=True).values.repeat(1,CH*H*W)
        Ytmp = torch.sgn(Ytmp) * self.relu(torch.abs(Ytmp) -th * Ym)
        return Ytmp.reshape(Y.shape)
    
        
    @property
    def layer_weights(self):
        return (self.convd.weight.data,self.conve.weight.data)

    @property
    def layer_thresholds(self):
        return (self.thL.item(),self.thR.item(),self.coef0.item())
    
    @property
    def layer_convd(self):
        return self.convd

 
class LCRPCA(nn.Module):
    def __init__(self,params=None):
        super().__init__()
        
        self.layers=params['layers']
        self.kernel=params['kernel']
        self.iter_weight_share = params['iter_weight_share']
        self.rk=params['rk']
        self.filter=self.makelayers()
        self.relu = nn.ReLU()

        if self.iter_weight_share:
            assert torch.allclose(self.filter[0].convd.weight.data,
                                  self.filter[-1].convd.weight.data),'iter_weight_share selected but not good'
            
        self.conv_detect= self.filter[-1].convd 
        
    def makelayers(self):
        if self.iter_weight_share:
            layer = ISTACell(1,1,self.kernel[0],self.kernel[1],self.rk)
            filt  = nn.Sequential(*[layer for _ in range(self.layers)])
        else:
            filt=[]
            for i in range(self.layers):
                filt.append(ISTACell(1,1,self.kernel[i][0],self.kernel[i][1],self.rk))
            filt = nn.Sequential(*filt) 
        
        return filt
    
    def forward(self,data):
        _,L,R=self.filter(data)
        Y = self.relu(L) + self.relu(self.filter[-1].layer_convd(R))
        R_detect= self.conv_detect(R).sigmoid().squeeze()
        return (Y,L,R,R_detect)
    
    @property
    def conv_dicts_weights(self):
        return [self.filter[i].layer_weights for i in range(self.layers)]
    
    @property 
    def thresholds(self):
        return [self.filter[i].layer_thresholds for i in range(self.layers)] 
    
    @property
    def features(self):
        return [self.filter[i].layer_features for i in range(self.layers)]
