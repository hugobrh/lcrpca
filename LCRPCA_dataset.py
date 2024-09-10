# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data

def preprocess(L,R,Y):
    
    A=max(torch.max(torch.abs(L)),torch.max(torch.abs(R)),torch.max(torch.abs(Y)))   
    
    if A==0:
        A=1
        
    L=L/A
    R=R/A
    Y=Y/A
                
    return L,R,Y
    
class SimuDataset(data.Dataset):

    def __init__(self, NumInstances, shape, train, transform=None, data_dir=None):
                
        data_dir = self.DATA_DIR if data_dir is None else data_dir
        self.shape=shape

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances])+(1,*self.shape))
        images_R = torch.zeros(tuple([NumInstances])+(1,*self.shape))
        images_Y = torch.zeros(tuple([NumInstances])+(1,*self.shape))
        images_Rtrue = torch.zeros(tuple([NumInstances])+(self.shape))
        images_ind = torch.zeros(tuple([NumInstances]))
    
        
        #   --  TRAIN  --
        if train == 0:
            for n in range(NumInstances):
                if n % 200 == 0: print('loading train set %s' % (n))
                
                sim_nb,noise_nb = n // 10, n % 10
                
                Y=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Ybp.npy'))
                L=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Lbp.npy'))
                R=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Rbp.npy'))
                Rtrue=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_R.npy'))
                Rtrue[Rtrue > 0.0] = 1.0
                
                L,R,Y=preprocess(L,R,Y)
                
                images_L[n] = L.reshape(1,*L.shape)
                images_R[n] = R.reshape(1,*R.shape)
                images_Y[n] = Y.reshape(1,*Y.shape)
                images_Rtrue[n] = Rtrue.reshape(*Rtrue.shape)
                images_ind[n] = sim_nb
                    
        #   --  VALIDATION --
        if train == 1:
            IndParam = 2400
            for n in range(IndParam, IndParam + NumInstances):
                if (n - IndParam) % 200 == 0: print('loading validation set %s' % (n - IndParam))
               
                sim_nb,noise_nb = n // 10, n % 10
                
                Y=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Ybp.npy'))
                L=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Lbp.npy'))
                R=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Rbp.npy'))
                Rtrue=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_R.npy'))
                Rtrue[Rtrue > 0.0] = 1.0

                L,R,Y=preprocess(L,R,Y)
                
                images_L[n-IndParam] = L.reshape(1,*L.shape)
                images_R[n-IndParam] = R.reshape(1,*R.shape)
                images_Y[n-IndParam] = Y.reshape(1,*Y.shape)
                images_Rtrue[n-IndParam] = Rtrue.reshape(*Rtrue.shape)
                images_ind[n-IndParam] = sim_nb
    
        
        #   --  TEST --
        if train == 2:
            IndParam = 3200
            for n in range(IndParam, IndParam + NumInstances):
                if (n - IndParam) % 10 == 0: print('loading test1 set %s' % (n - IndParam))
                
                sim_nb,noise_nb = n // 10, n % 10
                
                Y=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Ybp.npy'))
                L=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Lbp.npy'))
                R=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_noise_{noise_nb}_Rbp.npy'))
                Rtrue=torch.tensor(np.load(data_dir + f'scene_{sim_nb}_R.npy'))
                Rtrue[Rtrue > 0.0] = 1.0

                L,R,Y=preprocess(L,R,Y)
                
                images_L[n-IndParam] = L.reshape(1,*L.shape)
                images_R[n-IndParam] = R.reshape(1,*R.shape)
                images_Y[n-IndParam] = Y.reshape(1,*Y.shape)
                images_Rtrue[n-IndParam] = Rtrue.reshape(*Rtrue.shape)
                images_ind[n-IndParam] = sim_nb
                
   
        self.transform = transform
        self.images_L = images_L
        self.images_R = images_R
        self.images_Y = images_Y
        self.images_Rtrue = images_Rtrue
        self.images_ind = images_ind
        

    def __getitem__(self, index):
        Y = self.images_Y[index]
        L = self.images_L[index]
        R = self.images_R[index]
        Rtrue = self.images_Rtrue[index]
        ind = self.images_ind[index]
        
        if self.Yfft:
            Yf = self.images_Yf[index]

        if self.Yfft:
            return Y, Yf, L, R, Rtrue, ind
        else:
            return Y, L, R, Rtrue, ind
    
    def __len__(self):
        return len(self.images_L)
