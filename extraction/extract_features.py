import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse
import torchvision
from PIL import Image
import numpy as np
from extraction.pytorch_i3d import InceptionI3d #file
import pdb

def load_frame(frame_file):

	data = Image.open(frame_file)
	data = data.resize((340, 256), Image.ANTIALIAS)
	data = np.array(data)
	data = data.astype(float)
	data = (data * 2 / 255) - 1

	assert(data.max()<=1.0)
	assert(data.min()>=-1.0)

	return data


def oversample_data(data): # (39, 16, 224, 224, 2)  # Check twice

	data_flip = np.array(data[:,:,:,::-1,:])

	data_1 = np.array(data[:, :, :224, :224, :])
	data_2 = np.array(data[:, :, :224, -224:, :])
	data_3 = np.array(data[:, :, 16:240, 58:282, :])   # ,:,16:240,58:282,:
	data_4 = np.array(data[:, :, -224:, :224, :])
	data_5 = np.array(data[:, :, -224:, -224:, :])

	data_f_1 = np.array(data_flip[:, :, :224, :224, :])
	data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
	data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
	data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
	data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

	return [data_1, data_2, data_3, data_4, data_5,
		data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (256,340,3))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:] = load_frame(os.path.join(frames_dir+"/rgb", 
				rgb_files[frame_indices[i][j]]))

	return batch_data


def load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices):
	batch_data = np.zeros(frame_indices.shape + (256,340,2))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,0] = load_frame(os.path.join(frames_dir+"/flow_x", 
				flow_x_files[frame_indices[i][j]]))

			batch_data[i,j,:,:,1] = load_frame(os.path.join(frames_dir+"/flow_y", 
				flow_y_files[frame_indices[i][j]]))

	return batch_data


def run(mode, load_model, sample_mode, frequency, frames_dir, batch_size=40):
	chunk_size = 16

	assert(mode in ['rgb', 'flow'])
	assert(sample_mode in ['oversample', 'center_crop'])
	
	# setup the model
	if mode == 'flow':
		i3d = InceptionI3d(400, in_channels=2)
	else:
		i3d = InceptionI3d(400, in_channels=3)
	i3d.load_state_dict(torch.load(load_model))
	i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode

	def forward_batch(b_data):
		b_data = b_data.transpose([0, 4, 1, 2, 3])
		b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224

		b_data = Variable(b_data.cuda(), volatile=True).float()
		b_features = i3d.extract_features(b_data)
		
		b_features = b_features.data.cpu().numpy()[:,:,0,0,0]
		return b_features

	if mode == 'rgb':
		rgb_files = [i for i in os.listdir(frames_dir+"/rgb")]
		rgb_files.sort()
		frame_cnt = len(rgb_files)
	else:
		flow_x_files = [i for i in os.listdir(frames_dir+"/flow_x")]
		flow_y_files = [i for i in os.listdir(frames_dir+"/flow_y")]
		flow_x_files.sort()
		flow_y_files.sort()
		assert(len(flow_y_files) == len(flow_x_files))
		frame_cnt = len(flow_y_files)
	# clipped_length = (frame_cnt // chunk_size) * chunk_size   # Cut frames
	# Cut frames
	assert(frame_cnt > chunk_size)
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk
	frame_indices = [] # Frames to chunks
	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
	frame_indices = np.array(frame_indices)
	#frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]
	else:
		full_features = [[]]
	for batch_id in range(batch_num):
		if mode == 'rgb':    
			batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
		else:
			batch_data = load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices[batch_id])
		if sample_mode == 'oversample':
			batch_data_ten_crop = oversample_data(batch_data)
			for i in range(10):
				pdb.set_trace()
				assert(batch_data_ten_crop[i].shape[-2]==224)
				assert(batch_data_ten_crop[i].shape[-3]==224)
				full_features[i].append(forward_batch(batch_data_ten_crop[i]))
		else:
			if sample_mode == 'center_crop':
				batch_data = batch_data[:,:,16:240,58:282,:] # Center Crop  (39, 16, 224, 224, 2)
			
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			full_features[0].append(forward_batch(batch_data))

	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	return full_features
