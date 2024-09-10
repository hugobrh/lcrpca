# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as data
from LCRPCA_dataset import SimuDataset
from LCRPCA_network import LCRPCA
from misc import norm_nuc,norm_sparse, DiceLoss

import time, datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle

ProjectName = 'xxx'
prefix      = 'xxx'


"""Network Settings"""

nbfilt= 64
params_net={}
params_net['rk']= 3
params_net['layers']= 6
params_net['iter_weight_share']= False
params_net['init_mode']= 'naive'

if params_net['iter_weight_share'] == True :
    params_net['kernel']= (nbfilt,7) #(nb of filters, filter size: must be odd !)
elif params_net['iter_weight_share'] == False :
    params_net['kernel']= [(nbfilt,7)]*2 + [(nbfilt,5)]*2 + [(nbfilt,3)]*2

init_mode= params_net['init_mode']

TrainInstances = 2400 # Size of training dataset
ValInstances   = 800

BatchSize      = 10
ValBatchSize   = 10

minloss = torch.inf

#Loading data
print('Loading phase...')
print('----------------')

data_dir = "./gprmax/"

shape_dset=(110,100)

#training
train_dataset=SimuDataset(round(TrainInstances),shape_dset,
                           train=0,data_dir=data_dir)
train_loader=data.DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)


#validation
val_dataset=SimuDataset(round(ValInstances),shape_dset, 
                         train=1,data_dir=data_dir)
val_loader=data.DataLoader(val_dataset,batch_size=ValBatchSize,shuffle=True)

print('Finished loading.\n')

#%% Construct network        
    
alpha          = 1/3
beta           = 1/3

learning_rate  = 1e-3
num_epochs     = 50

lcrpca=LCRPCA(params_net)

assert torch.cuda.is_available(), "Cuda not available"
lcrpca = lcrpca.cuda()

    
#Loss and optimizer

# optimizer = torch.optim.SGD(lcrpca.parameters(),lr=learning_rate,
#                             momentum=0.9,nesterov=True,weight_decay=0.01)
optimizer = torch.optim.AdamW(lcrpca.parameters(),lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                steps_per_epoch=len(train_loader),
                                                epochs=num_epochs)

lossmean_vec=torch.zeros((num_epochs,4)) 
lossmean_val_vec=torch.zeros((num_epochs,4))


#%% Training loop

#torch.autograd.set_detect_anomaly(True)

print('Training the model over %d samples, with learning rate %.6f\n'\
      %(TrainInstances,learning_rate))

for epoch in range(num_epochs):
    print(f'epoch {epoch+1} / {num_epochs}')
    #print time
    ts=time.time()
    st=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print('\n'+st)
    
    regress_loss_val_mean=0
    regress_loss_mean=0
    
    regul_loss_val_mean=0
    regul_loss_mean=0

    detect_loss_val_mean=0
    detect_loss_mean=0

    loss_val_mean=0
    loss_mean=0
    
    # Train
    print('Loading and calculating training batches...')
    starttime=time.time()
    
    lcrpca.train()
    for num_batch,(Y,L,R,Rtrue,ind) in enumerate(train_loader):
        # set the gradients to zero at the beginning of each epoch
        optimizer.zero_grad()  
        
        noisy_Y = Y.cuda()
        true_R = Rtrue.cuda()
        
        if init_mode == 'rpca':
            init_L = L.cuda()
            init_R = R.repeat(1,nbfilt,1,1).cuda()
        elif init_mode == 'zero':  
            init_L = torch.zeros(noisy_Y.shape).cuda()
            init_R = torch.zeros(noisy_Y.shape).repeat(1,nbfilt,1,1).cuda()
        elif init_mode == 'naive':
            init_L = Y.cuda()
            init_R = torch.zeros(noisy_Y.shape).repeat(1,nbfilt,1,1).cuda()


        # Forward + backward + loss
        inputs = [noisy_Y,init_L,init_R]
        out_Y,out_L,out_R,out_R_detect=lcrpca(inputs)  # Forward
    
        # Current loss
        # delta = torch.median(torch.abs(out_Y-noisy_Y)).item()
        
        regress_loss = nn.MSELoss()(out_Y,noisy_Y)
        regul_loss   = (norm_nuc(out_L) + norm_sparse(out_R, mode='l1'))/out_L.numel()
        detect_loss  = nn.BCELoss()(out_R_detect,true_R) + DiceLoss()(out_R_detect,true_R) 
                
        #Total
        loss = alpha*regress_loss + (1-alpha-beta)*detect_loss
        
        regress_loss_mean += regress_loss.item()
        regul_loss_mean   += regul_loss.item()
        detect_loss_mean  += detect_loss.item()
        loss_mean         += loss.item()
        
        loss.backward()

        optimizer.step()
        
        scheduler.step()

        
        if num_batch % 10 == 0 :
            print(f'Batch {num_batch} / {round(TrainInstances/BatchSize)} done')
    
    regress_loss_mean = regress_loss_mean/TrainInstances
    regul_loss_mean   = regul_loss_mean/TrainInstances
    detect_loss_mean  = detect_loss_mean/TrainInstances
    loss_mean         = loss_mean/TrainInstances

    
    endtime=time.time()
    print('Training time is %f'%(endtime-starttime))
      
    # Validation 
    print('Loading and calculating validation batches...')
    starttime=time.time()
    
    lcrpca.eval()
    
    for num_batch,(Yv,Lv,Rv,Rvtrue,indv) in enumerate(val_loader): 
        noisy_Yv = Yv.cuda()# gprmax simu ground thruth
        true_Rv  = Rvtrue.cuda() # gprmax simu ground thruth

        if init_mode == 'rpca':
            init_Lv  = Lv.cuda()# rpca initialization  
            init_Rv  = Rv.repeat(1,nbfilt,1,1).cuda()  # rpca initialization 
        elif init_mode == 'zero':  
            init_Lv  = torch.zeros(noisy_Yv.shape).cuda()
            init_Rv= torch.zeros(noisy_Yv.shape).repeat(1,nbfilt,1,1).cuda()
        elif init_mode == 'naive':
            init_Lv = Yv.cuda()
            init_Rv= torch.zeros(noisy_Yv.shape).repeat(1,nbfilt,1,1).cuda()

        # Forward 
        inputsv = [noisy_Yv,init_Lv,init_Rv]
        out_Yv,out_Lv,out_Rv,out_Rv_detect=lcrpca(inputsv)     
        
        # Current loss
        # delta_v = torch.median(torch.abs(out_Yv-noisy_Yv)).item()
        
        regress_loss_v = nn.MSELoss()(out_Yv,noisy_Yv)
        regul_loss_v  = (norm_nuc(out_Lv) + norm_sparse(out_Rv,mode='l1'))/out_Lv.numel()  
        detect_loss_v = nn.BCELoss()(out_Rv_detect,true_Rv) + DiceLoss()(out_Rv_detect,true_Rv)
        
        #Total 
        loss_val = alpha*regress_loss_v + (1-alpha-beta)*detect_loss_v
        
        regress_loss_val_mean += regress_loss_v.item()
        regul_loss_val_mean   += regul_loss_v.item()
        detect_loss_val_mean  += detect_loss_v.item()
        loss_val_mean         += loss_val.item()
                             
        if num_batch % 10 == 0 :
            print(f'Batch {num_batch} / {round(ValInstances/ValBatchSize)} done')
           
    
    regress_loss_val_mean = regress_loss_val_mean/ValInstances
    regul_loss_val_mean   = regul_loss_val_mean/ValInstances
    detect_loss_val_mean  = detect_loss_val_mean/ValInstances
    loss_val_mean         = loss_val_mean/ValInstances

    
    endtime=time.time()
    print('Test time is %f'%(endtime-starttime))
    
    lossmean_vec[epoch,0]=regress_loss_mean
    lossmean_val_vec[epoch,0]=regress_loss_val_mean

    lossmean_vec[epoch,1]= regul_loss_mean
    lossmean_val_vec[epoch,1]= regul_loss_val_mean

    lossmean_vec[epoch,2]= detect_loss_mean
    lossmean_val_vec[epoch,2]= detect_loss_val_mean

    lossmean_vec[epoch,3]=loss_mean
    lossmean_val_vec[epoch,3]=loss_val_mean
            
    #Plot train components loss until this epoch
    epochs_vec=np.arange(0,num_epochs,1)

    plt.semilogy(epochs_vec,lossmean_vec[:,0],'-*',label='regress loss')
    plt.semilogy(epochs_vec,lossmean_vec[:,1],'-*',label='regul loss')
    plt.semilogy(epochs_vec,lossmean_vec[:,2],'-*',label='detect loss')

    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.title("Train components Losses")
    plt.legend()   
    plt.show()
    plt.close()
    plt.pause(0.0001) #Note this correction
    
    
    #Plot validation components loss until this epoch
    epochs_vec=np.arange(0,num_epochs,1)

    plt.semilogy(epochs_vec,lossmean_val_vec[:,0],'-*',label='regress loss val')
    plt.semilogy(epochs_vec,lossmean_val_vec[:,1],'-*',label='regul loss val')
    plt.semilogy(epochs_vec,lossmean_val_vec[:,2],'-*',label='detect loss val')

    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.title("Validation components Losses")
    plt.legend()   
    plt.show()
    plt.close()
    plt.pause(0.0001) #Note this correction
        
    #Plot total loss until this epoch
    epochs_vec=np.arange(0,num_epochs,1)
    plt.semilogy(epochs_vec,lossmean_vec[:,3],'-*',label='loss')
    plt.semilogy(epochs_vec,lossmean_val_vec[:,3],'-*',label='loss val')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.title("Total Loss")
    plt.legend()   
    plt.show()
    plt.close()
    plt.pause(0.0001) #Note this correction

    #Save model if validation loss is better
    
    if loss_val_mean < minloss:
        print('saved at [epoch%d/%d]'\
              %(epoch+1,num_epochs))
        torch.save(lcrpca.state_dict(), 
                   './Results/%s_%s_LCRPCA_Model_al%.2f_Tr%s_epoch%s_lr%.2e.pkl'\
                   %(ProjectName,prefix,alpha,TrainInstances,
                    num_epochs,learning_rate))
            
        minloss=min(loss_val_mean,minloss)  
        
#Reload best model
lcrpca = LCRPCA(params_net)
net_save_path = './Results/%s_%s_LCRPCA_Model_al%.2f_Tr%s_epoch%s_lr%.2e.pkl'\
%(ProjectName,prefix,alpha,TrainInstances,num_epochs,learning_rate)
net_state_dict=torch.load(net_save_path,map_location='cpu')
lcrpca.load_state_dict(net_state_dict)
assert torch.cuda.is_available(), "Cuda not available"
lcrpca = lcrpca.cuda()
    
    
#%% Test load

folder= "xxx"
directory = f'/xxx/{folder}/'

dim_scene = np.load(directory + 'dim_scene.npy')
dim_img = np.load(directory + 'dim_img.npy')
crossr_targs = np.load(directory + 'crossr_targs.npy')
downr_targs = np.load(directory + 'downr_targs.npy')
radius_targs = np.load(directory + 'radius_targs.npy')

try:
    assert (lcrpca is None) is False
    print('lcrpca already loaded')
except NameError:
    print('lcrpca not found. Loading it from memory')
    net_params = np.load('./Results/%s_%s_LCRPCA_Params_al%.2f_Tr%s_epoch%s_lr%.2e.npy'\
            %(ProjectName,prefix,alpha,TrainInstances,num_epochs,learning_rate),allow_pickle=True).item()
    lcrpca = LCRPCA(net_params)
    net_save_path = './Results/%s_%s_LCRPCA_Model_al%.2f_Tr%s_epoch%s_lr%.2e.pkl'\
    %(ProjectName,prefix,alpha,TrainInstances,num_epochs,learning_rate)
    net_state_dict=torch.load(net_save_path,map_location='cpu')
    lcrpca.load_state_dict(net_state_dict)
    lcrpca = lcrpca.cuda()

# prediction
lcrpca.eval()   # Set model to evaluate mode

TestInstances = 100
TestBatchSize = 2

test_dataset=SimuDataset(round(TestInstances),shape_dset, 
                         train=2,data_dir=directory)
test_loader=data.DataLoader(test_dataset,batch_size=TestBatchSize,
                       shuffle=True)

#%% Test run

(Y,L,R,Rtrue,ind) = next(iter(test_loader))

Ynoisy = Y.cuda()
Lrpca, Rrpca = L.cuda(), R.cuda()

if init_mode == 'rpca':
    init_L = Lrpca
    init_R = Rrpca.repeat(1,nbfilt,1,1)
elif init_mode == 'zero':  
    init_L = torch.zeros(Ynoisy.shape).cuda()    
    init_R = torch.zeros(Ynoisy.shape).repeat(1,nbfilt,1,1).cuda()
elif init_mode == 'naive':
    init_L = Y.cuda()
    init_R = torch.zeros(Ynoisy.shape).repeat(1,nbfilt,1,1).cuda()



Yhat,Lhat,Rhat,Rdetect_hat = lcrpca([Ynoisy,init_L,init_R])