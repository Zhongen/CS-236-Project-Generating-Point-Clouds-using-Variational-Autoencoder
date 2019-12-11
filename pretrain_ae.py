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
parser.add_argument('--ind',  type=int, default=2, help="experiment index")
parser.add_argument('--cuda', type=int, default=0, help="GPU index")
parser.add_argument('--name', type=str, default='car', help="category")
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
dir_config='./config_all/config_ae/config_exp_'+str(args.ind)+'.json'
with open(dir_config, 'r') as handle:
    config = json.load(handle)
if "name" in config.keys():
	del config["name"]
print('*'*20+'Configuration'+'*'*20)
print('*'*20+'Category: '+args.name+'*'*20)
print(json.dumps(config,indent=4))
locals().update(config)

dir_save_model='./trained_models/pretrained_ae/'+args.name+'_exp_'+str(args.ind)+'/'
dir_log='./trained_models/pretrained_ae/result.csv'
#dir_fig='./fig_all/fig_ae/fig_exp_'+str(args.ind)+'/'
# if not os.path.exists(dir_fig):
#     os.makedirs(dir_fig)
if not os.path.exists(dir_save_model):
    os.makedirs(dir_save_model)

############################################
 # initialize models and optimizer
############################################
# encoder: PointNet for representation
# input to PointNet has dim: batch_size,3,N
model_pointnet=PointNetfeat().to(device)
# decompress PointNet representation
model_pointnet_compression=FeatureCompression(dim_rep=dim_rep).to(device)

# decoder
model_decoder_ae=MLP(dim_in=dim_rep,dim_out=3*N_hold,
                     num_hidden_layer=num_hidden_layer_decoder_ae,
                     width=width_decoder_ae).to(device)

dict_model={'PointNet':model_pointnet,
           'FeatureCompression':model_pointnet_compression,
           'DecoderAE':model_decoder_ae}

lst_model=nn.ModuleList(dict_model.values())
lst_model_name=list(dict_model.keys())
optimizer = optim.Adam(lst_model.parameters(), lr=lr)

############################################
# read data
############################################
dir_data='/home/yeye/bin/2/project/data/shape_net_core_uniform_samples_2048/'
with open('../data/dict_info.pkl', "rb") as handle:
    dict_info = pickle.load(handle)
dataset_train=ut.SamplePointCouldDataset(dir_data,dict_info[args.name],P,K,N_hold,\
        N_eval,'train',device=device)

############################################
 # train
############################################

with tqdm.tqdm(total=iter_max_ae) as pbar:
    for i in range(iter_max_ae):
        optimizer.zero_grad()

        X_hold,X_eval=dataset_train[0] #random sample each call
        X_hold=X_hold.squeeze(1) #batch_size,N_hold,3
        #X_eval=X_eval.squeeze(1) #batch_size,N_eval,3

        #extract set representation
        out,_,_=dict_model['PointNet'](X_hold.permute(0,2,1)) #out: batch, 1024
        set_rep=dict_model['FeatureCompression'](out) #set_rep: batch, dim_rep

        X_rec=dict_model['DecoderAE'](set_rep).reshape(P,N_hold,3)

        dist_1,dist_2 = chamfer_dist(X_hold, X_rec)
        loss = (torch.mean(dist_1)) + (torch.mean(dist_2))

        loss.backward()
        #ut.clip_grad(lst_model,5)
        optimizer.step()
        
        pbar.set_postfix(loss='{:.2e}'.format(loss.item()))
        pbar.update(1)

############################################
 # save model and log result
############################################
print('*'*20+'Saving models and logging results'+'*'*20)
for model_name,model in dict_model.items():
    path=dir_save_model+args.name+'_'+model_name+'.pkl'
    ut.save_model(model,path)

lst_result=[args.ind,args.name,'{:.2e}'.format(loss.item())]
ut.write_list_to_csv(lst_result,dir_log)