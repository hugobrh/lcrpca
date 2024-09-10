#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:38:56 2023

@author: hugobrehier
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.constants import c
from utils.delay_propagation import delai_propag_ttw

#Backprojection algorithm

def back_projection(y,dt,dim_scene,dx,dz,dn,xn, d, eps, zoff,posinitx=0.,rets=None,pulse=None,theta = np.pi,freespace=False):
    
    N = y.shape[1]
    
    if pulse is None:
        y_mf = y
    else:
        y_mf=np.zeros(y.shape,dtype=np.complex_)
        for n in range(N):
            y_mf[:,n]=fft.ifft(fft.fft(y[:,n]) * np.conj(fft.fft(pulse)))
      
    # plt.imshow(10*np.log10(np.abs(y_mf)+1e-18),aspect='auto',vmin=-50,origin='lower')
    # plt.colorbar()
    # plt.show()
         
    x=np.arange(0,dim_scene[0],dx)
    z=np.arange(0,dim_scene[1],dz)
    lx=len(x)
    lz=len(z)
    cimg=np.zeros([lz,lx],dtype=np.complex_)
    
    for xi in range(lx):
        for zi in range(lz):
            larg = 2*np.tan(theta/2)*z[zi]
            xmin = x[xi]- larg/2 - posinitx
            xmax = x[xi]+ larg/2 - posinitx
            
            try:
                recmin = np.where(xn >= xmin)[0][0]
            except IndexError:
                recmin = 0
              
            try:
                recmax = np.where(xn <= xmax)[0][-1]
            except IndexError:
                recmax = N
                                        
            for n in range(recmin,recmax):
                if rets is not None:
                    ret = rets[xi,zi,n]
                else:
                    if freespace:
                        ret=2*np.sqrt((x[xi]-xn[n])**2+(z[zi]-0.1)**2)/c
                    else:
                        if z[zi] <= zoff+d:
                            ret=2*np.sqrt((x[xi]-xn[n])**2+(z[zi])**2)/c
                        else:
                            try:
                                _,ret = delai_propag_ttw((x[xi],z[zi]),(xn[n],0), d, eps, zoff,halley=False)
                                #sprint("delay calculation successful")
                            except RuntimeError:
                                print("delay calculation failed")
                                print(f'target at : {(x[xi],z[zi])} . Radar at {(xn[n],0)}')
                               
                
                
                indice=int(round(ret/dt))
                cimg[zi,xi] += y_mf[indice,n]         
    return cimg

from matplotlib.patches import Circle

def plot_bp(img,dim_scene,dim_img,targets_crossr,targets_downr,targets_radius,decim=10,
            flip=True,title='Img',save_loc=None,color_bar=True):
    
    dx_bp,dz_bp = dim_scene/dim_img
    fig, ax = plt.subplots(figsize=(8,8))
    
    if flip:
        im = ax.imshow(img/np.max(img),aspect='equal',origin='lower',cmap='viridis',
                       extent = [0,dim_scene[0],0,dim_scene[1]])
    else:
        im = ax.imshow(img/np.max(img),aspect='equal',origin='upper',cmap='viridis',
                       extent = [0,dim_scene[0],0,dim_scene[1]])

    ax.set_xticks(np.arange(0,dim_scene[0]+dx_bp, decim*dx_bp))
    ax.set_yticks(np.arange(0,dim_scene[1]+dz_bp, decim*dz_bp))
    
    for i in range(len(targets_crossr)):
        circ = Circle(xy=(targets_crossr[i],targets_downr[i]), radius=targets_radius[i],
                      linewidth=1.61, edgecolor='r', facecolor='none')
        ax.add_patch(circ)
    
    ax.set_title(title)
    
    if color_bar:
        fig.colorbar(im)
        
    if save_loc is not None : 
        plt.savefig(save_loc)
    
    plt.show()
    
    
    
