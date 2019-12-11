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

#read arg
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ind_ae',  type=int, default=8, help="pretrained AE index")
parser.add_argument('--ind',  type=int, default=8, help="experiment index")
parser.add_argument('--cuda', type=int, default=0, help="GPU index")
parser.add_argument('--name', type=str, default='car', help="category")
parser.add_argument('--batch_size', type=int, default=100, help="batch size")

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

dir_load_model='./trained_models/vae/'+args.name+'_exp_'+str(args.ind)+'/'
dir_log='./trained_models/summary_result.csv'
############################################
 # initialize models
############################################
print('*'*20+'Initializing and loading models'+'*'*20)

z_prior_m=(torch.nn.Parameter(torch.zeros(1), requires_grad=False)+z_prior_m).to(device)
z_prior_v=(torch.nn.Parameter(torch.ones(1), requires_grad=False)*z_prior_v).to(device)

# encoder: PointNet for representation
# input to PointNet has dim: batch_size,3,N
model_pointnet=PointNetfeat().to(device)
# decompress PointNet representation
model_pointnet_compression=FeatureCompression(dim_rep=dim_rep).to(device)
# decoder
model_decoder_ae=MLP(dim_in=dim_rep,dim_out=3*N_hold,
                     num_hidden_layer=num_hidden_layer_decoder_ae,
                     width=width_decoder_ae).to(device)
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
#load traiend models
model_pointnet=ut.load_model(model_pointnet,dir_load_model+args.name+'_PointNet.pkl')
model_pointnet_compression=ut.load_model(model_pointnet_compression,dir_load_model+args.name+'_FeatureCompression.pkl')
model_decoder_ae=ut.load_model(model_decoder_ae,dir_load_model+args.name+'_DecoderAE.pkl')

model_z_to_rep=ut.load_model(model_z_to_rep,dir_load_model+args.name+'_LatentToRep.pkl')
model_encoder_vae=ut.load_model(model_encoder_vae,dir_load_model+args.name+'_EncoderVAE.pkl')

dict_model={'PointNet':model_pointnet,
           'FeatureCompression':model_pointnet_compression,
           'DecoderAE':model_decoder_ae,
           'EncoderVAE':model_encoder_vae,
           'LatentToRep':model_z_to_rep}

lst_model=nn.ModuleList(dict_model.values())
lst_model_name=list(dict_model.keys())


############################################
# read data
############################################
dir_data='/home/yeye/bin/2/project/data/shape_net_core_uniform_samples_2048/'
with open('../data/dict_info.pkl', "rb") as handle:
    dict_info = pickle.load(handle)

def eval(dataloader):
	lst_loss_vae=list()
	lst_loss_ae=list()

	for name_model,model in dict_model.items():
	    model.eval()
	with torch.no_grad():
	    for ind_batch, X_batch in enumerate(dataloader):
	        cur_batch_size=X_batch.shape[0]
	        #print(X_batch.shape)
	        #reconstruct
	        out,_,_=dict_model['PointNet'](X_batch.permute(0,2,1)) #out: batch, 1024
	        set_rep=dict_model['FeatureCompression'](out) #set_rep: batch, dim_rep

	        #encoding. dim: batch, dim_z
	        qm,qv=ut.gaussian_parameters(dict_model['EncoderVAE'](set_rep),dim=1)
	        #sample z
	        z=ut.sample_gaussian(qm,qv,device=device) #batch_size, dim_z
	        #z to rep
	        rep_m,rep_v=ut.gaussian_parameters(dict_model['LatentToRep'](z))
	        X_rec=dict_model['DecoderAE'](ut.sample_gaussian(rep_m,rep_v,device=device)).reshape(cur_batch_size,-1,3)

	        #ae
	        X_rec_ae=dict_model['DecoderAE'](set_rep).reshape(cur_batch_size,-1,3)

	        dist_1,dist_2 = chamfer_dist(X_batch, X_rec)
	        loss_vae = (torch.mean(dist_1,axis=1)) + (torch.mean(dist_2,axis=1))

	        dist_1,dist_2 = chamfer_dist(X_batch, X_rec_ae)
	        loss_ae = (torch.mean(dist_1,axis=1)) + (torch.mean(dist_2,axis=1))

	        lst_loss_vae.append(loss_vae)
	        lst_loss_ae.append(loss_ae)
	avg_loss_vae=torch.cat(lst_loss_vae).mean().item()
	avg_loss_ae=torch.cat(lst_loss_ae).mean().item()
	return avg_loss_vae, avg_loss_ae

dataset_test=ut.IteratePointCouldDataset(dir_data,dict_info[args.name],'valtest',N=N_hold,device=device)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=False)

dataset_train=ut.IteratePointCouldDataset(dir_data,dict_info[args.name],'train',N=N_hold,device=device)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=False)

#test loss
print('*'*20+'Evaluating test performance'+'*'*20)
avg_loss_vae_test,avg_loss_ae_test=eval(dataloader_test)
#train loss
print('*'*20+'Evaluating train performance'+'*'*20)
avg_loss_vae_train,avg_loss_ae_train=eval(dataloader_train)

############################################
 # log result
############################################
lst_result=[args.ind,args.name,'VAE',avg_loss_vae_train,avg_loss_vae_test]
ut.write_list_to_csv(lst_result,dir_log)

lst_result=[args.ind_ae,args.name,'AE',avg_loss_ae_train,avg_loss_ae_test]
ut.write_list_to_csv(lst_result,dir_log)

