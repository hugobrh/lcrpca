#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to create a dataset of GprMax simulations
# i.e. this package below to be installed
# https://docs.gprmax.com/en/latest/include_readme.html


from scipy.constants import c
import subprocess
import h5py
import cv2
import numpy as np
import os # adding this line
import matplotlib.pyplot as plt
from utils.backprojection import back_projection,plot_bp
from utils.rpca import R_pca
from utils.delay_propagation import delai_propag_ttw


os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

#To adapt to your setup
repo_name ='lcrpca_github'
folder= 'gprmax'
directory = f'./{folder}/'

location_in = directory + 'dataset_creation.in'
location_sh = directory + 'dataset_creation.sh'
      

#%% Grid creation

np.random.seed(7)

infile = open(location_in,"r")
inlines = infile.readlines()
infile.close()

shfile = open(location_sh,"r")
shlines = shfile.readlines()
shfile.close()

for i,line in enumerate(inlines):
    if "#box:" in line and ('wall' in line) or ('placo' in line) : 
        wall = line.split()
    if "#domain:" in line:
        domain = line.split()
    if "#src_steps:" in line:
        radar_step = float(line.split()[1])
    if "#rx:" in line:
        radar_pos_init = float(line.split()[1])
    if "#waveform:" in line:
        fc =  float(line.split()[3])
    if "#material:" in line:
        material = line.split()

    
ds_sim = float(domain[3]) #should be less than 0.1 * c/f_max = lambda_min/10
assert ds_sim <= 0.1*c/5e9

dt = 1/(c*np.linalg.norm([1/ds_sim,1/ds_sim]))


for i,line in enumerate(shlines):
    if '-gpu' in line : 
        npos = int(line.split()[5])

crossr_spacing = 0.5
downr_spacing  = 0.5

crossr = np.linspace(crossr_spacing, float(domain[1])-crossr_spacing, int(float(domain[1])/crossr_spacing))
downr  = np.linspace(float(wall[5])+downr_spacing, float(domain[2])-downr_spacing, int(float(domain[2])/downr_spacing))
crossrv, downrv = np.meshgrid(crossr, downr)

plt.plot(crossrv, downrv, marker='o', color='k', linestyle='none')
plt.show()


max_targs = 3
nsim  = crossrv.size * max_targs
ntargs = np.random.randint(1,max_targs+1,size=nsim)

radius_targs = np.random.randint(5,10,size=[max_targs,nsim])/100

crossr_targs = np.zeros([max_targs,nsim])
downr_targs = np.zeros([max_targs,nsim])

for i in range(max_targs):
    crossr_targs[i,:] = np.random.choice(crossr,size=nsim)
    downr_targs[i,:] = np.random.choice(downr,size=nsim)

for targs_nb in range(1,max_targs+1):
    crossr_targs[targs_nb:,np.where(ntargs == targs_nb)] = 0
    downr_targs[targs_nb:,np.where(ntargs == targs_nb)] = 0

radius_targs[np.where(crossr_targs==0)] = 0

np.save(directory + 'crossr_targs.npy', crossr_targs)
np.save(directory + 'downr_targs.npy', downr_targs)
np.save(directory + 'radius_targs.npy', radius_targs)

dx_bp,dz_bp = np.array([0.05,0.05])
dim_scene = (float(domain[1]),float(domain[2]))
dim_img = (dim_scene / np.array([dx_bp,dz_bp])).astype('int')
xn = np.linspace(radar_pos_init,radar_pos_init + radar_step*npos-radar_step,npos)


np.save(directory + 'dim_scene.npy', dim_scene)
np.save(directory + 'dim_img.npy', dim_img)


nb_noise = 10

d = float(wall[5]) - float(wall[2])
eps = float(material[1])
zoff = float(wall[2]) - 0.1
ricker = np.load(directory+ 'ricker.npy')

#Save ttw returns grid
try:
    rets = np.load(directory + 'rets.npy')
except FileNotFoundError:
    print('Rets save not found.Creating it...')

    rets_x=np.arange(0,dim_scene[0],dx_bp)
    rets_z=np.arange(0,dim_scene[1],dz_bp)
    rets = np.zeros((len(rets_x),len(rets_z),len(xn)))
    
    for xi in range(rets.shape[0]):
        for zi in range(rets.shape[1]):
            for n in range(rets.shape[2]):
                if rets_z[zi] <= zoff+d:
                    rets[xi,zi,n]=2*np.sqrt((rets_x[xi]-xn[n])**2+(rets_z[zi])**2)/c
                else:
                    _,rets[xi,zi,n] = delai_propag_ttw((rets_x[xi],rets_z[zi]),(xn[n],0),
                                                       d, eps, zoff,halley=False)
    np.save(directory+ 'rets.npy', rets)
    print('Rets created and saved.')

#%% Run simulation

try:
    tracking = np.load(directory + 'tracking.npy')
except FileNotFoundError:
    print('Tracking save not found.Initializing it...')
    tracking = np.zeros(nsim,dtype='bool')

for sim_nb,sim_done in enumerate(tracking):
    if sim_done == False:
        
        #Load template
        infile = open(location_in,"r")
        inlines = infile.readlines()
        infile.close()
        
        shfile = open(location_sh,"r")
        shlines = shfile.readlines()
        shfile.close()

        #Info
        perc_done = 100*np.sum(tracking)/np.size(tracking)
        print(f'{perc_done:.1f} % done.')

        #create scene ground truth ie R
        fig = plt.figure()
        ax = fig.add_axes([0.,0.,1.,1.])
        
        fig.set_size_inches((float(domain[1]),float(domain[2])))
        ax.set_xlim(0, float(domain[1]))
        ax.set_ylim(0, float(domain[2]))
        
        #ax.add_patch(plt.Rectangle((0,float(wall[2])), float(domain[1]), float(wall[5]) - float(wall[2])))
        for targ in range(np.count_nonzero(crossr_targs[:,sim_nb])):
            ax.add_patch(plt.Circle((crossr_targs[targ,sim_nb],downr_targs[targ,sim_nb]),
                                    radius_targs[targ,sim_nb],color='black'))
        
        ax.set_xticks(np.linspace(0,float(domain[1]),int(float(domain[1])/0.5)+1))
        ax.set_yticks(np.linspace(0,float(domain[2]),int(float(domain[2])/0.5)+1)) 

        ax.axis('off')
        fig.add_axes(ax)
        fig.canvas.draw()
        
        # this rasterized the figure
        X = np.array(fig.canvas.renderer._renderer)
        X = 0.2989*X[:,:,1] + 0.5870*X[:,:,2] + 0.1140*X[:,:,3]
        
        plt.close()
        
        Xb = np.zeros(X.shape)
        Xb[np.where(X<254.974)] = 1
        Xb_ds = cv2.resize(Xb,dim_img)
        np.save(directory+f'scene_{sim_nb}_R',Xb_ds)
        
        plot_bp(np.flip(Xb_ds,0), dim_scene, dim_img,crossr_targs[:,sim_nb],
                    downr_targs[:,sim_nb],radius_targs[:,sim_nb])

        #Edit in file
        for k,line in enumerate(inlines):
            if "#cylinder:" in line:
                inlines[k] = f'#cylinder: {crossr_targs[0,sim_nb]:.1f} {downr_targs[0,sim_nb]:.1f} 0 {crossr_targs[0,sim_nb]:.1f} {downr_targs[0,sim_nb]:.1f} {domain[3]} {radius_targs[0,sim_nb]} pec \n'
                
                if ntargs[sim_nb] >= 2 :
                    for targ_nb in range(1,ntargs[sim_nb]):
                        inlines.insert(k+targ_nb,f'#cylinder: {crossr_targs[targ_nb,sim_nb]:.1f} {downr_targs[targ_nb,sim_nb]:.1f} 0 {crossr_targs[targ_nb,sim_nb]:.1f} {downr_targs[targ_nb,sim_nb]:.1f} {domain[3]} {radius_targs[targ_nb,sim_nb]} pec \n')
                break #needed as the insert will create an infinte loop
        
        file = open(directory + f'scene_{sim_nb}.in',"w")
        file.writelines(inlines)
        file.close()
        
        #Edit sh file
        for k,line in enumerate(shlines):
            if "gpu" in line:
                shlines[k] = f'python -m gprMax home/xxx/Documents/{repo_name}/{directory}/scene_{sim_nb}.in -n {npos} -gpu \n'
            if "outputfiles_merge" in line:
                shlines[k] = f'python -m tools.outputfiles_merge home/xxx/Documents/{repo_name}/{directory}/scene_{sim_nb} --remove-files \n'

        file = open(directory + f'scene_{sim_nb}.sh',"w")
        file.writelines(shlines)
        file.close()
              
        #Run bash file containing the GprMax simulation script
        subprocess.run(["chmod", "+x",directory+ f'scene_{sim_nb}.sh'])
        subprocess.call(directory+f'scene_{sim_nb}.sh',
                        shell=True,executable="/bin/bash")
        
        #create the Bscan (ie Y)
        f = h5py.File(directory+ f'scene_{sim_nb}_merged.out', 'r')
        dt = f.attrs["dt"]
        Bscan = np.array(f['rxs']["rx1"]["Ez"])
        f.close()
        
        ''' 
        itse = Bscan.shape[0] 
        gprmax_dir = './data_sample/'
        pulse_filename = 'ricker.txt'
        pulse = np.loadtxt(gprmax_dir+pulse_filename,skiprows=1)
        embed_pulse = np.zeros(itse)
        embed_pulse[:len(pulse)] = pulse
        embed_pulse_fft = np.fft.fft(embed_pulse)
    
        '''
        
        t_crosstalk = 2*(float(wall[2])-0.1)/c
        its_crosstalk = int(t_crosstalk/dt)
        Bscan[0:its_crosstalk,:] = 0
        
        plt.imshow(10*np.log10(np.abs(Bscan)+1e-18),aspect='auto',vmin=-50,origin='lower')
        plt.colorbar()
        plt.show()
        np.save(directory+f'scene_{sim_nb}_Y',Bscan)
                    
        #create the back projection image with noise
        
        for some_noise in range(nb_noise):            
            #multivariate student-t noise via coumpojnd gaussin characterisation
            
            snr_db  = np.random.randint(10,30)
            snr_lin = 10**(snr_db/10)
            t_df = np.round(np.random.uniform(2.5,5),1)
            
            sp = np.mean(Bscan**2) # Signal Power
            std_n = np.sqrt(sp / snr_lin) # Noise std. deviation

            cgn = std_n * np.random.randn(*Bscan.shape)
            x = np.random.gamma(shape=t_df/2,scale=2,size= Bscan.shape[1])
            text = t_df/x
            Bscan_n = Bscan + cgn @ np.diag(np.sqrt(text))

            bp_img = back_projection(Bscan_n, dt, dim_scene, dx_bp, dz_bp, radar_step,
                                     xn,d,eps,zoff,rets, pulse=ricker,freespace=False,posinitx=radar_pos_init)
            plot_bp(np.abs(bp_img), dim_scene, dim_img,crossr_targs[:,sim_nb],
                    downr_targs[:,sim_nb],radius_targs[:,sim_nb])

            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Ybp',np.flip(np.abs(bp_img),0))
        
            #create the FFT of the Bscans
            itse = Bscan_n.shape[0] 
            freqs = np.fft.fftfreq(n=itse,d=dt)[:itse//2]
            select_freqs = (freqs>=1e9)&(freqs<=3e9)
            
            Bscan_fft = np.array([np.fft.fft(Bscan_n[:,i]) for i in range(Bscan_n.shape[1])]).T
            
            '''
            Bscan_fft = np.array([np.fft.fft(Bscan_n[:,i]) * np.conj(embed_pulse_fft)
                                  for i in range(Bscan_n.shape[1])]).T
            '''
            
            Bscan_fft_curated = np.array([Bscan_fft[:,i][:itse//2][select_freqs] 
                                          for i in range(Bscan_fft.shape[1])]).T
            plt.imshow(np.abs(Bscan_fft_curated),aspect='auto')
            plt.show()
            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Yfft',Bscan_fft_curated)

        #Mark as done and save progression
        tracking[sim_nb] = True
        np.save(directory + 'tracking.npy', tracking)
                

#%% BP sample test

from utils.backprojection import back_projection,plot_bp

ind = 5
Bscan_n = np.load(directory + f'scene_{ind}_noise_{5}_Ybp.npy')

ricker = np.load(directory+'ricker.npy')

d = float(wall[5]) - float(wall[2])
eps = float(material[1])
zoff = float(wall[2]) - 0.1

bp_img = back_projection(Bscan_n, dt, dim_scene, dx_bp, dz_bp, radar_step, xn,d,eps,zoff,
                         rets=rets,pulse=ricker,posinitx=radar_pos_init)
plot_bp(bp_img, dim_scene, dim_img,crossr_targs[:,ind],downr_targs[:,ind],radius_targs[:,ind])

L,S = R_pca(bp_img).fit()
plot_bp(L, dim_scene, dim_img)
plot_bp(S, dim_scene, dim_img)

#%% init for adding things after simu already done 

'''
with os.scandir(directory) as thisdir:
    for entry in thisdir:
        if entry.is_file() and 'noise' in entry.name:
            os.remove(entry.path)
'''      

np.random.seed(7)

infile = open(location_in,"r")
inlines = infile.readlines()
infile.close()

shfile = open(location_sh,"r")
shlines = shfile.readlines()
shfile.close()

for i,line in enumerate(inlines):
    if "#box:" in line and ('wall' in line) or ('placo' in line) : 
        wall = line.split()
    if "#domain:" in line:
        domain = line.split()
    if "#src_steps:" in line:
        radar_step = float(line.split()[1])
    if "#rx:" in line:
        radar_pos_init = float(line.split()[1])
    if "#waveform:" in line:
        fc =  float(line.split()[3])
    if "#material:" in line:
        material = line.split()

    
ds_sim = float(domain[3]) #should be less than 0.1 * c/f_max = lambda_min/10
assert ds_sim <= 0.1*c/5e9

dt = 1/(c*np.linalg.norm([1/ds_sim,1/ds_sim]))

for i,line in enumerate(shlines):
    if '-gpu' in line : 
        npos = int(line.split()[5])
        
dx_bp,dz_bp = np.array([0.05,0.05])
dim_scene = (float(domain[1]),float(domain[2]))
dim_img = (dim_scene / np.array([dx_bp,dz_bp])).astype('int')
xn = np.linspace(radar_pos_init,radar_pos_init + radar_step*npos-radar_step,npos)


nb_noise = 10

d = float(wall[5]) - float(wall[2])
eps = float(material[1])
zoff = float(wall[2]) - 0.1

ricker = np.load(directory+'/ricker.npy')
rets = np.load(directory + 'rets.npy')
dim_scene = np.load(directory + 'dim_scene.npy')
dim_img = np.load(directory + 'dim_img.npy')
crossr_targs = np.load(directory + 'crossr_targs.npy')
downr_targs = np.load(directory + 'downr_targs.npy')
radius_targs = np.load(directory + 'radius_targs.npy')
nsim = crossr_targs.shape[1]

#%% add RPCA to dataset after simu already done

try:
    tracking2 = np.load(directory + 'tracking2.npy')
except FileNotFoundError:
    print('Tracking save not found.Initializing it...')
    tracking2 = np.zeros(nsim,dtype='bool')

for sim_nb,sim_done in enumerate(tracking2):
    if sim_done == False:
        for some_noise in range(nb_noise):
            print(f'Scene {sim_nb}...')
            Y = np.load(directory + f'scene_{sim_nb}_noise_{some_noise}_Ybp.npy')
            L,S = R_pca(Y).fit(max_iter=500)
            
            # plot_bp(L, dim_scene, dim_img,
            #         crossr_targs[:,sim_nb],downr_targs[:,sim_nb],radius_targs[:,sim_nb],
            #         flip=False)
            # plot_bp(S, dim_scene, dim_img,
            #         crossr_targs[:,sim_nb],downr_targs[:,sim_nb],radius_targs[:,sim_nb],
            #         flip=False)
            
            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Lbp',L)
            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Rbp',S)
    
        tracking2[sim_nb] = True
        np.save(directory + 'tracking2.npy', tracking2)
        
#%% add BP with noise to dataset after simu already done

try:
    tracking3 = np.load(directory + 'tracking3.npy')
except FileNotFoundError:
    print('Tracking save not found.Initializing it...')
    tracking3 = np.zeros(nsim,dtype='bool')


for sim_nb,sim_done in enumerate(tracking3):
    if sim_nb==0:
        f = h5py.File(directory+ f'scene_{sim_nb}_merged.out', 'r')
        dt = f.attrs["dt"]
        f.close()
    
    if sim_done == False:
        print(f'Scene {sim_nb}...')
        Bscan = np.load(directory+f'scene_{sim_nb}_Y.npy')
        
        for some_noise in range(nb_noise):
            #create the back projection image with noise
            snr_db  = np.random.randint(4,8) #(10,30)
            snr_lin = 10**(snr_db/10)
            t_df = np.random.uniform(2,3) #(2.5,5)

            sp = np.mean(Bscan**2) # Signal Power
            std_n = np.sqrt(sp / snr_lin) # Noise std. deviation
            
            cgn = std_n * np.random.randn(*Bscan.shape)
            x = np.random.gamma(shape=t_df/2,scale=2,size= Bscan.shape[1])
            text = t_df/x
            Bscan_n = Bscan + cgn @ np.diag(np.sqrt(text))

            bp_img = back_projection(Bscan_n, dt, dim_scene, dx_bp, dz_bp, radar_step,
                                     xn,d,eps,zoff,rets, pulse=ricker,freespace=False)
            # plot_bp(np.abs(bp_img), dim_scene, dim_img,crossr_targs[:,sim_nb],
            #         downr_targs[:,sim_nb],radius_targs[:,sim_nb])

            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Ybp',np.flip(np.abs(bp_img),0))
            
            #create the FFT of the Bscans
            itse = Bscan_n.shape[0] 
            freqs = np.fft.fftfreq(n=itse,d=dt)[:itse//2]
            select_freqs = (freqs>=1e9)&(freqs<=3e9)
            Bscan_fft = np.array([np.fft.fft(Bscan_n[:,i]) for i in range(Bscan_n.shape[1])]).T
            Bscan_fft_curated = np.array([Bscan_fft[:,i][:itse//2][select_freqs] 
                                          for i in range(Bscan_fft.shape[1])]).T
            # plt.imshow(np.abs(Bscan_fft_curated),aspect='auto')
            # plt.show()
            np.save(directory+f'scene_{sim_nb}_noise_{some_noise}_Yfft',Bscan_fft_curated)

        tracking3[sim_nb] = True
        np.save(directory + 'tracking3.npy', tracking3)
