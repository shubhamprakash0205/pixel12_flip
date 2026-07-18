import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pickle
import os 
from PIL import Image
import matplotlib.pyplot as plt
from model import sevenpixels,image_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_transform = transforms.Compose([
	transforms.ToTensor()
])
grd_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(p=1.0)  
])	

myimg_dataset = image_dataset(grd_transform = grd_transform)
myimg_dataloader = DataLoader(myimg_dataset,shuffle=True,batch_size=128*40)

epoch_count = 0				
mypixel_model = sevenpixels().to(device)
optimizer = optim.Adam(mypixel_model.parameters(),lr=1e-5)
loss_func = nn.MSELoss()


if os.path.exists('files/result/flip_checkpoint.pth'):
	checkpoint = torch.load('files/result/flip_checkpoint.pth',map_location=torch.device(device))
	mypixel_model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])		
	epoch_count = checkpoint['epoch_count']
	prev_loss = checkpoint['loss']
	for param_group in optimizer.param_groups:
		param_group['lr'] = 1e-5
	


def update_process():
	global input_i,output_o
	xloss_sum = 0
	for batch in myimg_dataloader:
		input_i,output_o = batch
		input_i,output_o = input_i.to(device),output_o.to(device)
		prediction_o = mypixel_model(input_i)
		#print(prediction_o.shape,output_o.shape)
		xloss = loss_func(prediction_o,output_o)
		optimizer.zero_grad()
		xloss.backward()
		optimizer.step()
		
		#print(mypixel_model.layer1_z1_R.weight[0])
		#print('####')
		xloss_sum = xloss_sum + xloss * input_i.size(0)
		
	xloss_sum = torch.sum(xloss_sum) / len(myimg_dataset)
	#print(f'epoch_count: {epoch_count}')
	if epoch_count % 1 == 0:
		print(f'epoch_count: {epoch_count}')
		torch.save({'epoch_count':epoch_count,'model_state_dict':mypixel_model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':xloss_sum},'files/result/flip_checkpoint.pth')
		print('saved ....')
		print(xloss_sum.item(),'LOSS')
	return xloss_sum
		
		
def train():
	global epoch_count
	prev_loss = 0
	while True:
		current_loss = update_process()
		epoch_count += 1
		#if abs(current_loss-prev_loss) > 1e-50:
		if epoch_count < 550000:	
			prev_loss = current_loss
		else:
			print('Finisheddd')
			break
	print(prev_loss,current_loss)			
		

	
train()





