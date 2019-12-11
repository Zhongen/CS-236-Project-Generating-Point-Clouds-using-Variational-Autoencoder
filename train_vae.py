import argparse
import os
import numpy as np
import pickle
import json
import tqdm
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import utils as ut
from PointNet import *
from nns import *
from pyTorchChamferDistance.chamfer_distance.chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#from tensorboardX import SummaryWriter

#read arg
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ind_ae',  type=int, default=2, help="pretrained AE index")
parser.add_argument('--ind',  type=int, default=1, help="experiment index")
parser.add_argument('--cuda', type=int, default=-1, help="GPU index")
parser.add_argument('--name', type=str, default='car', help="category")
#parser.add_argument('--freeze_ae', type=int, default=0, help="whether to freeze weights in ae while training vae")

args = parser.parse_args()

############################################
# local GPU setup
############################################
if args.cuda==-1:
    print('use CPU')
    device = torch.device('cpu')
else:
    cuda_visible_gpu=[args.cuda]
    device_name='cuda'
    #device_name='cuda:'+str(args.cuda)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in cuda_visible_gpu)
    print('visible device: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    print('number of gpu visible: {}'.format(torch.cuda.device_count()))
    print('torch cuda version: {}'.format(torch.version.cuda))
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

############################################
# read config file
############################################
dir_config_ae='./config_all/config_ae/config_exp_'+str(args.ind_ae)+'.json'
dir_config_vae='./config_all/config_vae/config_exp_'+str(args.ind)+'.json'
with open(dir_config_ae, 'r') as handle:
    config_ae = json.load(handle)
with open(dir_config_vae, 'r') as handle:
    config_vae = json.load(handle)

config=config_ae.copy()
config.update(config_vae)
if "name" in config.keys():
    del config["name"]

print('*'*20+'Configuration'+'*'*20)
print('*'*20+'Category: '+args.name+'*'*20)
print(json.dumps(config,indent=4))
locals().update(config)

dir_pretrained_ae='./trained_models/pretrained_ae/'+args.name+'_exp_'+str(args.ind_ae)+'/'
dir_save_model='./trained_models/vae/'+args.name+'_exp_'+str(args.ind)+'/'
dir_log='./trained_models/vae/result.csv'
dir_fig='./fig_all/fig_vae/'+args.name+'_exp_'+str(args.ind)+'/'

if not os.path.exists(dir_fig):
    os.makedirs(dir_fig)
if not os.path.exists(dir_save_model):
    os.makedirs(dir_save_model)

############################################
 # initialize models and optimizer
############################################
print('*'*20+'Initializing and loading models'+'*'*20)
# encoder: PointNet for representation
# input to PointNet has dim: batch_size,3,N
model_pointnet=PointNetfeat().to(device)
# decompress PointNet representation
model_pointnet_compression=FeatureCompression(dim_rep=dim_rep).to(device)
# decoder
model_decoder_ae=MLP(dim_in=dim_rep,dim_out=3*N_hold,
                     num_hidden_layer=num_hidden_layer_decoder_ae,
                     width=width_decoder_ae).to(device)

#load pretrained ae
model_pointnet=ut.load_model(model_pointnet,dir_pretrained_ae+args.name+'_PointNet.pkl')
model_pointnet_compression=ut.load_model(model_pointnet_compression,dir_pretrained_ae+args.name+'_FeatureCompression.pkl')
model_decoder_ae=ut.load_model(model_decoder_ae,dir_pretrained_ae+args.name+'_DecoderAE.pkl')

#initialize VAE models
#VAE encoder
model_encoder_vae=MLP(dim_in=dim_rep,
                  dim_out=dim_z*2,
                  num_hidden_layer=num_hidden_layer_encoder_vae,
                  width=width_encoder_vae).to(device)
#VAE latent variable to compact representation, output gaussian params
model_z_to_rep=MLP(dim_in=dim_z,
                  dim_out=dim_rep*2,
                  num_hidden_layer=num_hidden_layer_z_to_rep,
                  width=width_z_to_rep).to(device)

dict_model={'PointNet':model_pointnet,
           'FeatureCompression':model_pointnet_compression,
           'DecoderAE':model_decoder_ae,
           'EncoderVAE':model_encoder_vae,
           'LatentToRep':model_z_to_rep}

lst_model=nn.ModuleList(dict_model.values())
lst_model_name=list(dict_model.keys())
optimizer = optim.Adam(lst_model.parameters(), lr=lr)

# freeze parameters 
if freeze_ae==1:
    for param in dict_model['DecoderAE'].parameters():
        param.requires_grad = False
    for param in dict_model['PointNet'].parameters():
        param.requires_grad = False
    for param in dict_model['FeatureCompression'].parameters():
        param.requires_grad = False

############################################
# read data
############################################
dir_data='/home/yeye/bin/2/project/data/shape_net_core_uniform_samples_2048/'
with open('../data/dict_info.pkl', "rb") as handle:
    dict_info = pickle.load(handle)
dataset_train=ut.SamplePointCouldDataset(dir_data,dict_info[args.name],P,K,N_hold,N_eval,'train',device=device)

############################################
 # train
############################################
z_prior_m=(torch.nn.Parameter(torch.zeros(1), requires_grad=False)+z_prior_m).to(device)
z_prior_v=(torch.nn.Parameter(torch.ones(1), requires_grad=False)*z_prior_v).to(device)
with tqdm.tqdm(total=iter_max) as pbar:
    for i in range(iter_max):
        optimizer.zero_grad()
        X_hold,_=dataset_train[0] #random sample each call
        X_hold=X_hold.squeeze(1) #batch_size,N_hold,3
        #X_eval=X_eval.squeeze(1) #batch_size,N_eval,3

        #extract set representation from hold out set
        out,_,_=dict_model['PointNet'](X_hold.permute(0,2,1)) #out: batch, 1024
        set_rep=dict_model['FeatureCompression'](out) #set_rep: batch, dim_rep

        #encoding. dim: batch, dim_z
        qm,qv=ut.gaussian_parameters(dict_model['EncoderVAE'](set_rep),dim=1)
        #sample z
        z=ut.sample_gaussian(qm,qv,device=device) #batch_size, dim_z
        #z to rep
        rep_m,rep_v=ut.gaussian_parameters(dict_model['LatentToRep'](z))

        log_likelihood=ut.log_normal(set_rep,rep_m,rep_v) #dim: batch
        lb_1=log_likelihood.mean() #scalar

        #KL divergence
        m=z_prior_m.expand(P,dim_z)
        v=z_prior_v.expand(P,dim_z)
        lb_2=-ut.kl_normal(qm,qv,m,v).mean() #scalar

        loss=-1*(lb_1+lb_2)
        loss.backward()
        optimizer.step()

        #reconstruct and plot
        if i%iter_rec==0:
            X_rec=dict_model['DecoderAE'](ut.sample_gaussian(rep_m,rep_v,device=device)).reshape(P,-1,3)
            for j in range(5):
	            fig=ut.plot_3d_point_cloud(X_hold[j,:,:].detach().cpu().numpy(),
	                X_rec[j,:,:].detach().cpu().numpy());
	            fig.savefig(dir_fig+args.name+'_iter_'+str(i)+'_pic_'+str(j+1)+'.png')
	            plt.close(fig)
                
        pbar.set_postfix(
            loss='{:.2e}'.format(loss.item()),
            kl='{:.2e}'.format(lb_2.item()),
            rec='{:.2e}'.format(lb_1.item()))
        pbar.update(1)

############################################
 # save model and log result
############################################
print('*'*20+'Saving models and logging results'+'*'*20)
for model_name,model in dict_model.items():
    path=dir_save_model+args.name+'_'+model_name+'.pkl'
    ut.save_model(model,path)

lst_result=[args.ind,args.ind_ae,args.name,'{:.2e}'.format(loss.item())]
ut.write_list_to_csv(lst_result,dir_log)