import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import os 
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



class image_dataset(Dataset):
	def __init__(self,input_transform=None,grd_transform = None):
		img_folder_path = 'files/train'
		file_list = os.listdir(os.path.join(img_folder_path))
		self.file_list = [os.path.join(img_folder_path,x) for x in file_list if os.path.isfile(os.path.join(img_folder_path,x))]
		print(self.file_list)
		self.dict_list =[]
		self.label_list = []
		
		for f in self.file_list:
			with open(f,'rb') as file:
				dict = pickle.load(file,encoding='bytes')
				print(dict.keys())
				#print(dict[b'filenames'])
				x = dict[b'data'] #contains numpy array of shape(10,000 * 3072)
				y = dict[b'labels'] #list of 10,000 integers
				
				self.dict_list.append(x)
				self.label_list.append(y)
		
		#self.all_img_data becomes 50,000 * 3072 each row one img data and total there are 50,000 images
		#self.all_labels is a list of 50,000 integers 
		self.all_img_data = np.concatenate((self.dict_list[0],self.dict_list[1],self.dict_list[2],self.dict_list[3],self.dict_list[4]),axis=0)						
		self.all_img_data = self.all_img_data / 255
		self.all_labels = [*self.label_list[0],*self.label_list[1],*self.label_list[2],*self.label_list[3],*self.label_list[4]]
		print(self.all_img_data.shape)
		print(len(self.all_labels))
			
		self.input_transform = input_transform
		self.grd_transform = grd_transform
	
		
		
	def __len__(self):
		return len(self.all_labels)
		
	
	def __getitem__(self,idx):
		img = self.all_img_data[idx]
		grd = self.all_img_data[idx]
		
		img = torch.tensor(img,dtype=torch.float32)
		grd = torch.tensor(grd,dtype=torch.float32)
		img = img.reshape(3,32,32)
		grd = grd.reshape(3,32,32)
		
		if self.grd_transform:
			grd = self.grd_transform(grd)
		
		return img,grd


class sevenpixels(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.layer1_z1_R = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer1_z1_G = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias =True,padding=3)	
		self.layer1_z1_B = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias = True,padding=3)
				
		self.layer2_z2_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer2_z2_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer2_z2_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer3_z3_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer3_z3_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer3_z3_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		
		
		self.layer4_z4_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer4_z4_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer4_z4_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		
		
		self.layer5_z5_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer5_z5_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer5_z5_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3) 
		
		
		self.layer6_z6_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer6_z6_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer6_z6_B = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer7_z7_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer7_z7_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer7_z7_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer8_z8_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer8_z8_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer8_z8_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer9_z9_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer9_z9_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer9_z9_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer10_z10_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer10_z10_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer10_z10_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		
		
		self.layer11_z11_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer11_z11_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer11_z11_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		
		self.layer_activation = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer12_z12_R = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size=7,bias=True,padding=3)
		self.layer12_z12_G = nn.Conv2d(in_channels=7,out_channels =1,kernel_size=7,bias=True,padding=3)
		self.layer12_z12_B = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size=7,bias=True,padding=3)
		
		self.layer12_activation = nn.Sigmoid()
		

		
	def forward(self,input_image):
		#input_image = [1,3,h,w]
		input_image_r = input_image[:,:1,:,:]
		input_image_g = input_image[:,1:2,:,:]
		input_image_b = input_image[:,2:3,:,:]
		
		layer1_o_r = self.layer_activation(self.layer1_z1_R(input_image_r)) #(batch_size,7,h,w)
		layer1_o_g = self.layer_activation(self.layer1_z1_G(input_image_g))
		layer1_o_b = self.layer_activation(self.layer1_z1_B(input_image_b))
		
		layer2_o_r = self.layer_activation(self.layer2_z2_R(layer1_o_r))
		layer2_o_g = self.layer_activation(self.layer2_z2_G(layer1_o_g))
		layer2_o_b = self.layer_activation(self.layer2_z2_B(layer1_o_b))
		
		layer3_o_r = self.layer_activation(self.layer3_z3_R(layer2_o_r))
		layer3_o_g = self.layer_activation(self.layer3_z3_G(layer2_o_g))
		layer3_o_b = self.layer_activation(self.layer3_z3_B(layer2_o_b))
		
		layer4_o_r = self.layer_activation(self.layer4_z4_R(layer3_o_r))
		layer4_o_g = self.layer_activation(self.layer4_z4_G(layer3_o_g))
		layer4_o_b = self.layer_activation(self.layer4_z4_B(layer3_o_b))
		
		layer5_o_r = self.layer_activation(self.layer5_z5_R(layer4_o_r))
		layer5_o_g = self.layer_activation(self.layer5_z5_G(layer4_o_g))
		layer5_o_b = self.layer_activation(self.layer5_z5_B(layer4_o_b))
		
		layer6_o_r = self.layer_activation(self.layer6_z6_R(layer5_o_r))
		layer6_o_g = self.layer_activation(self.layer6_z6_G(layer5_o_g))
		layer6_o_b = self.layer_activation(self.layer6_z6_B(layer5_o_b))
		
		#print(layer6_o_r.shape) (batch_size,7,H,W)
		
		layer7_o_r = self.layer_activation(self.layer7_z7_R(layer6_o_r))
		layer7_o_g = self.layer_activation(self.layer7_z7_G(layer6_o_g))
		layer7_o_b = self.layer_activation(self.layer7_z7_B(layer6_o_b))
		
		layer8_o_r = self.layer_activation(self.layer8_z8_R(layer7_o_r))
		layer8_o_g = self.layer_activation(self.layer8_z8_G(layer7_o_g))
		layer8_o_b = self.layer_activation(self.layer8_z8_B(layer7_o_b))
		
		layer9_o_r = self.layer_activation(self.layer9_z9_R(layer8_o_r))
		layer9_o_g = self.layer_activation(self.layer9_z9_G(layer8_o_g))
		layer9_o_b = self.layer_activation(self.layer9_z9_B(layer8_o_b))
		
		layer10_o_r = self.layer_activation(self.layer10_z10_R(layer9_o_r))
		layer10_o_g = self.layer_activation(self.layer10_z10_G(layer9_o_g))
		layer10_o_b = self.layer_activation(self.layer10_z10_B(layer9_o_b))

		layer11_o_r = self.layer_activation(self.layer11_z11_R(layer10_o_r))
		layer11_o_g = self.layer_activation(self.layer11_z11_G(layer10_o_g))
		layer11_o_b = self.layer_activation(self.layer11_z11_B(layer10_o_b))

		layer12_o_r = self.layer12_activation(self.layer12_z12_R(layer11_o_r))
		layer12_o_g = self.layer12_activation(self.layer12_z12_G(layer11_o_g))
		layer12_o_b = self.layer12_activation(self.layer12_z12_B(layer11_o_b))
		
		#print(layer12_o_r.shape) #(batch_size,1,H,W)
					
		predicted_o = torch.concatenate((layer12_o_r,layer12_o_g,layer12_o_b),axis = 1)
		#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
		
		return predicted_o





		
		





