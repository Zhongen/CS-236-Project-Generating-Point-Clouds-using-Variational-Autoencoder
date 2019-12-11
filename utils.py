import numpy as np
import os
import shutil
import sys
import torch
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from src.in_out import load_ply

import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

class IteratePointCouldDataset(Dataset):
	def __init__(self,dir_data,dict_cat,option,N=1024,device='cpu'):
		'''
		- Iterate though the entire dataset, return N samples from the entire point cloud
		- dict_cat only correspond to ONE category, e.g. dict_cat=dict_info['table']
		- load all data when initialized
		'''
		assert option in ['train','val','test','valtest']
		self.cid=dict_cat['cid']
		self.name=dict_cat['name']
		self.device=device
		self.N=N

		print('*'*10+'IteratePointCouldDataset: Loading '+option+' data for '+\
			self.name+'*'*10)

		#list of all ply files
		if option !='valtest':
			lst_dir_ply=dict_cat['dict_split'][option]
		else:
			lst_dir_ply_test=dict_cat['dict_split']['test']
			lst_dir_ply_val=dict_cat['dict_split']['val']
			lst_dir_ply=lst_dir_ply_test+lst_dir_ply_val
		self.num_plot=len(lst_dir_ply)
		#load all items
		self.lst_ply=list()
		for filename in lst_dir_ply:
			dir_ply=dir_data+self.cid+'/'+filename
			arr_ply=load_ply(dir_ply)
			arr_ply=torch.tensor(arr_ply).float()
			self.lst_ply.append(arr_ply)

	def get_cid(self):
		return self.cid
	def get_name(self):
		return self.name
	def __len__(self):
		return len(self.lst_ply)
	def __getitem__(self,i):
		ind_point=np.random.choice(2048,self.N)
		arr_ply=self.lst_ply[i].index_select(0,torch.LongTensor(ind_point))
		return arr_ply.to(self.device)

class SampleOnePlotDataset(Dataset):
	def __init__(self,dir_data,dict_cat,P,K,N_hold,N_eval,option,ind_plot=0,device='cpu'):
		'''
		- only sample point clouds from ONE PLOT
		'''
		assert option in ['train','val','test']
		self.P, self.K, self.N_eval, self.N_hold=P,K,N_eval,N_hold
		self.cid=dict_cat['cid']
		self.name=dict_cat['name']
		self.device=device
		#list of all ply files
		lst_dir_ply=dict_cat['dict_split'][option]
		filename=lst_dir_ply[ind_plot]
		dir_ply=dir_data+self.cid+'/'+filename
		arr_ply=load_ply(dir_ply)
		self.arr_ply=torch.tensor(arr_ply).float()

	def __getitem__(self,i):
		'''
		- random sampling for each call.
		- same i will give different samples 
		'''
		lst_ply_hold_temp,lst_ply_eval_temp=list(),list()
		for k in range(self.K):
			ind_point=np.random.choice(2048,self.N_eval+self.N_hold)
			ply=self.arr_ply.index_select(0,torch.LongTensor(ind_point))
			lst_ply_hold_temp.append(ply[:self.N_hold,:])
			lst_ply_eval_temp.append(ply[self.N_hold:,:])
		arr_ply_hold=torch.stack(lst_ply_hold_temp).unsqueeze(0) # (1, K, N_hold, 3)
		arr_ply_eval=torch.stack(lst_ply_eval_temp).unsqueeze(0) # (1, K, N_eval, 3)
		return arr_ply_hold.to(self.device),arr_ply_eval.to(self.device)


class SamplePointCouldDataset(Dataset):
	def __init__(self,dir_data,dict_cat,P,K,N_hold,N_eval,option,device='cpu'):
		'''
		- Sample P plots. For each plot, sample K times (density eval, hold out)
		- N_eval and N_hold are sizes for density_eval and hold_out
		- dict_cat only correspond to ONE category, e.g. dict_cat=dict_info['table']
		- load all data when initialized
		- dataset[i] will be different every time it is called
		- no need for loader
		- send data to device when __getitem__ is called
		'''
		assert option in ['train','val','test']
		self.P, self.K, self.N_eval, self.N_hold=P,K,N_eval,N_hold
		self.cid=dict_cat['cid']
		self.name=dict_cat['name']
		self.device=device
		#list of all ply files
		lst_dir_ply=dict_cat['dict_split'][option]
		self.num_plot=len(lst_dir_ply)
		#load all items
		self.lst_ply=list()
		
		print('*'*10+'SamplePointCouldDataset: Loading '+option+' data for '+\
			self.name+'*'*10)
		for filename in lst_dir_ply:
			dir_ply=dir_data+self.cid+'/'+filename
			arr_ply=load_ply(dir_ply)
			arr_ply=torch.tensor(arr_ply).float()
			self.lst_ply.append(arr_ply)
	def get_cid(self):
		return self.cid
	def get_name(self):
		return self.name
	def __getitem__(self,i):
		'''
		- random sampling for each call.
		- same i will give different samples 
		'''
		#sample P plots out of num_plot plots
		lst_ind_plot=np.random.choice(self.num_plot,self.P)
		#sample 
		lst_ply_hold,lst_ply_eval=list(),list()
		for ind_p in lst_ind_plot:
			lst_ply_hold_temp,lst_ply_eval_temp=list(),list()
			for k in range(self.K):
				ind_point=np.random.choice(2048,self.N_eval+self.N_hold)
				ply=self.lst_ply[ind_p].index_select(0,torch.LongTensor(ind_point))
				lst_ply_hold_temp.append(ply[:self.N_hold,:])
				lst_ply_eval_temp.append(ply[self.N_hold:,:])
			lst_ply_hold.append(torch.stack(lst_ply_hold_temp))
			lst_ply_eval.append(torch.stack(lst_ply_eval_temp))
		arr_ply_hold=torch.stack(lst_ply_hold) # (P, K, N_hold, 3)
		arr_ply_eval=torch.stack(lst_ply_eval) # (P, K, N_eval, 3)
		return arr_ply_hold.to(self.device),arr_ply_eval.to(self.device)

class SamplePointCouldDatasetOnFly(Dataset):
    def __init__(self,dir_data,dict_cat,P,K,N_hold,N_eval,option,device='cpu'):
        '''
        - load data when returning items, not when dataset is initialized
        - Sample P plots. For each plot, sample K times (density eval, hold out)
        - N_eval and N_hold are sizes for density_eval and hold_out
        - dict_cat only correspond to ONE category, e.g. dict_cat=dict_info['table']
        - dataset[i] will be different every time it is called
        - no need for loader
        - send data to device when __getitem__ is called
        '''
        assert option in ['train','val','test']
        self.P, self.K, self.N_eval, self.N_hold=P,K,N_eval,N_hold
        self.cid=dict_cat['cid']
        self.name=dict_cat['name']
        self.device=device
        #list of all ply files
        self.lst_dir_ply=dict_cat['dict_split'][option]
        self.num_plot=len(self.lst_dir_ply)
        self.dir_data=dir_data
        
        print('*'*10+'SamplePointCouldDatasetOnFly: Loading '+option+' data for '+\
            self.name+'*'*10)

    def get_cid(self):
        return self.cid
    def get_name(self):
        return self.name
    def __getitem__(self,i):
        '''
        - random sampling for each call.
        - same i will give different samples 
        '''
        #sample P plots out of num_plot plots
        lst_ind_plot=np.random.choice(self.num_plot,self.P)
        #sample 
        lst_ply_hold,lst_ply_eval=list(),list()
        for ind_p in lst_ind_plot:
            lst_ply_hold_temp,lst_ply_eval_temp=list(),list()
            dir_ply=self.dir_data+self.cid+'/'+self.lst_dir_ply[ind_p]
            arr_ply=torch.tensor(load_ply(dir_ply)).float()
            
            for k in range(self.K):
                ind_point=np.random.choice(2048,self.N_eval+self.N_hold)
                ply=arr_ply.index_select(0,torch.LongTensor(ind_point))

                lst_ply_hold_temp.append(ply[:self.N_hold,:])
                lst_ply_eval_temp.append(ply[self.N_hold:,:])
            lst_ply_hold.append(torch.stack(lst_ply_hold_temp))
            lst_ply_eval.append(torch.stack(lst_ply_eval_temp))
        arr_ply_hold=torch.stack(lst_ply_hold) # (P, K, N_hold, 3)
        arr_ply_eval=torch.stack(lst_ply_eval) # (P, K, N_eval, 3)
        return arr_ply_hold.to(self.device),arr_ply_eval.to(self.device)
        
def clip_grad(model, max_norm):
	total_norm = 0
	for p in model.parameters():
		if p.requires_grad:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm ** 2
	total_norm = total_norm ** (0.5)
	clip_coef = max_norm / (total_norm + 1e-6)
	if clip_coef < 1:
		for p in model.parameters():
			if p.requires_grad:
				p.grad.data.mul_(clip_coef)
	return total_norm

def sample_gaussian(m, v,device='cpu'):
	"""
	Element-wise application reparameterization trick to sample from Gaussian

	Args:
		m: tensor: (batch, ...): Mean
		v: tensor: (batch, ...): Variance

	Return:
		z: tensor: (batch, ...): Samples
	"""
	z=torch.mul(torch.randn(m.shape).to(device),torch.sqrt(v))+m
	return z

def log_normal(x, m, v):
	"""
	Computes the elem-wise log probability of a Gaussian and then sum over the
	last dim. Basically we're assuming all dims are batch dims except for the
	last dim.

	Args:
		x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
		m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
		v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

	Return:
		log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
			each sample. Note that the summation dimension is not kept
	"""
	dim=m.shape[-1]
	const=-dim/2*np.log(2*np.pi)
	log_prob=const-0.5*torch.log(v.sum(axis=-1))-0.5*((x-m)**2/v).sum(axis=-1)
	return log_prob

def gaussian_parameters(h, dim=-1):
	"""
	Converts generic real-valued representations into mean and variance
	parameters of a Gaussian distribution

	Args:
		h: tensor: (batch, ..., dim, ...): Arbitrary tensor
		dim: int: (): Dimension along which to split the tensor for mean and
			variance

	Returns:
		m: tensor: (batch, ..., dim / 2, ...): Mean
		v: tensor: (batch, ..., dim / 2, ...): Variance
	"""
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	return kl

def duplicate(x, rep):
	"""
	Duplicates x along dim=0

	Args:
		x: tensor: (batch, ...): Arbitrary tensor
		rep: int: (): Number of replicates. Setting rep=1 returns orignal x

	Returns:
		_: tensor: (batch * rep, ...): Arbitrary replicated tensor
	"""
	return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

def save_model_by_name(model, global_step,label=None):
	if label is None:
		save_dir = os.path.join('checkpoints', model.name)
	else:
		save_dir = os.path.join('checkpoints', model.name,label)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))

def load_model_by_name(model, global_step, device=None,label=None):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	if label is None:
		file_path = os.path.join('checkpoints',model.name,\
			'model-{:05d}.pt'.format(global_step))
	else:
		file_path = os.path.join('checkpoints',model.name,label,\
			'model-{:05d}.pt'.format(global_step))
	state = torch.load(file_path, map_location=device)
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))

def plot_3d_point_cloud(X_hold,X_sampled, show=True, show_axis=True, \
                    in_u_sphere=False, marker='.', s=8, alpha=.8, \
                    figsize=(5, 5), elev=10, azim=240, axis=None, title=None, *args, **kwargs):

    '''
    input X_hold and X_samples are 2D np arrays
    '''
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc_1 = ax.scatter(X_hold[:,0], X_hold[:,1], X_hold[:,2], marker=marker, s=s, \
                    alpha=alpha,c='b', *args, **kwargs)
    sc_2 = ax.scatter(X_sampled[:,0], X_sampled[:,1], X_sampled[:,2], marker=marker, s=s, \
                    alpha=alpha,c='r', *args, **kwargs)
    
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        X=np.concatenate((X_sampled,X_hold))
        miv = 0.7 * np.min([np.min(X[:,0]), np.min(X[:,1]), np.min(X[:,2])])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(X[:,0]), np.max(X[:,1]), np.max(X[:,2])])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc_1)

    if show:
        plt.show()

    return fig
def save_model(model,path):
    torch.save(model.state_dict(),path)
    
def load_model(model_container,path):
    model_container.load_state_dict(torch.load(path))
    return model_container

def write_list_to_csv(lst,dir_csv):
    lst.append('\n')
    row=','.join([str(i) for i in lst])
    with open(dir_csv,'a') as fd:
        fd.write(row)