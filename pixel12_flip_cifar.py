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

input_transform = transforms.Compose([
	transforms.ToTensor()
])

grd_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(p=1.0)  
])	

myimg_dataset = image_dataset(grd_transform = grd_transform)
myimg_dataloader = DataLoader(myimg_dataset,shuffle=True,batch_size=128*40)

"""
print(myimg_dataset[0][1].shape)
plt.subplot(1,1,1)
plt.imshow(myimg_dataset[0][1])
plt.title('is_it_flipped')
plt.show()
"""


class sevenpixels(nn.Module):
	def __init__(self):
		super().__init__()

		self.layer1_z1_R = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer1_a1_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer1_z1_G = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer1_a1_G = nn.LeakyReLU(negative_slope=1e-4)		
		
		self.layer1_z1_B = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=7,bias = True,padding=3)
		self.layer1_a1_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer2_a2_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer2_a2_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer2_a2_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer3_a3_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer3_a3_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer3_a3_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer4_a4_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer4_a4_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer4_a4_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer5_a5_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias =True,padding=3)
		self.layer5_a5_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3) 
		self.layer5_a5_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer6_a6_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=7,bias=True,padding=3)
		self.layer6_a6_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_B = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer6_a6_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer7_a7_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer7_a7_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer7_a7_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer8_z8_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer8_a8_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer8_z8_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer8_a8_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer8_z8_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer8_a8_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer9_z9_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer9_a9_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer9_z9_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer9_a9_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer9_z9_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer9_a9_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer10_z10_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer10_a10_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer10_z10_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer10_a10_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer10_z10_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer10_a10_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer11_z11_R = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer11_a11_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer11_z11_G = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=7,bias=True,padding=3)
		self.layer11_a11_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer11_z11_B = nn.Conv2d(in_channels=7,out_channels= 7,kernel_size=7,bias=True,padding=3)
		self.layer11_a11_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer12_z12_R = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size=7,bias=True,padding=3)
		self.layer12_a12_R = nn.Sigmoid()
		
		self.layer12_z12_G = nn.Conv2d(in_channels=7,out_channels =1,kernel_size=7,bias=True,padding=3)
		self.layer12_a12_G = nn.Sigmoid()
		
		self.layer12_z12_B = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size=7,bias=True,padding=3)
		self.layer12_a12_B = nn.Sigmoid()
		

		
	def forward(self,input_image):
		#input_image = [1,3,h,w]
		input_image_r = input_image[:,:1,:,:]
		input_image_g = input_image[:,1:2,:,:]
		input_image_b = input_image[:,2:3,:,:]
		
		layer1_o_r = self.layer1_a1_R(self.layer1_z1_R(input_image_r)) #(batch_size,7,h,w)
		layer1_o_g = self.layer1_a1_G(self.layer1_z1_G(input_image_g))
		layer1_o_b = self.layer1_a1_B(self.layer1_z1_B(input_image_b))
		
		layer2_o_r = self.layer2_a2_R(self.layer2_z2_R(layer1_o_r))
		layer2_o_g = self.layer2_a2_G(self.layer2_z2_G(layer1_o_g))
		layer2_o_b = self.layer2_a2_B(self.layer2_z2_B(layer1_o_b))
		
		layer3_o_r = self.layer3_a3_R(self.layer3_z3_R(layer2_o_r))
		layer3_o_g = self.layer3_a3_G(self.layer3_z3_G(layer2_o_g))
		layer3_o_b = self.layer3_a3_B(self.layer3_z3_B(layer2_o_b))
		
		layer4_o_r = self.layer4_a4_R(self.layer4_z4_R(layer3_o_r))
		layer4_o_g = self.layer4_a4_G(self.layer4_z4_G(layer3_o_g))
		layer4_o_b = self.layer4_a4_B(self.layer4_z4_B(layer3_o_b))
		
		layer5_o_r = self.layer5_a5_R(self.layer5_z5_R(layer4_o_r))
		layer5_o_g = self.layer5_a5_G(self.layer5_z5_G(layer4_o_g))
		layer5_o_b = self.layer5_a5_B(self.layer5_z5_B(layer4_o_b))
		
		layer6_o_r = self.layer6_a6_R(self.layer6_z6_R(layer5_o_r))
		layer6_o_g = self.layer6_a6_G(self.layer6_z6_G(layer5_o_g))
		layer6_o_b = self.layer6_a6_B(self.layer6_z6_B(layer5_o_b))
		
		#print(layer6_o_r.shape) (batch_size,7,H,W)
		
		layer7_o_r = self.layer7_a7_R(self.layer7_z7_R(layer6_o_r))
		layer7_o_g = self.layer7_a7_G(self.layer7_z7_G(layer6_o_g))
		layer7_o_b = self.layer7_a7_B(self.layer7_z7_B(layer6_o_b))
		
		layer8_o_r = self.layer8_a8_R(self.layer8_z8_R(layer7_o_r))
		layer8_o_g = self.layer8_a8_G(self.layer8_z8_G(layer7_o_g))
		layer8_o_b = self.layer8_a8_B(self.layer8_z8_B(layer7_o_b))
		
		layer9_o_r = self.layer9_a9_R(self.layer9_z9_R(layer8_o_r))
		layer9_o_g = self.layer9_a9_G(self.layer9_z9_G(layer8_o_g))
		layer9_o_b = self.layer9_a9_B(self.layer9_z9_B(layer8_o_b))
		
		layer10_o_r = self.layer10_a10_R(self.layer10_z10_R(layer9_o_r))
		layer10_o_g = self.layer10_a10_G(self.layer10_z10_G(layer9_o_g))
		layer10_o_b = self.layer10_a10_B(self.layer10_z10_B(layer9_o_b))

		layer11_o_r = self.layer11_a11_R(self.layer11_z11_R(layer10_o_r))
		layer11_o_g = self.layer11_a11_G(self.layer11_z11_G(layer10_o_g))
		layer11_o_b = self.layer11_a11_B(self.layer11_z11_B(layer10_o_b))

		layer12_o_r = self.layer12_a12_R(self.layer12_z12_R(layer11_o_r))
		layer12_o_g = self.layer12_a12_G(self.layer12_z12_G(layer11_o_g))
		layer12_o_b = self.layer12_a12_B(self.layer12_z12_B(layer11_o_b))
		
		#print(layer12_o_r.shape) #(batch_size,1,H,W)
					
		predicted_o = torch.concatenate((layer12_o_r,layer12_o_g,layer12_o_b),axis = 1)
		#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
		
		return predicted_o
		

epoch_count = 0		
		
mypixel_model = sevenpixels().to(device)
optimizer = optim.Adam(mypixel_model.parameters(),lr=1e-4)
loss_func = nn.MSELoss()


if os.path.exists('files/result/flip_checkpoint.pth'):
	checkpoint = torch.load('files/result/flip_checkpoint.pth',map_location=torch.device(device))
	mypixel_model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])		
	epoch_count = checkpoint['epoch_count']
	prev_loss = checkpoint['loss']




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
		print(xloss_sum,'LOSS')
	return xloss_sum
		
		
def main():
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
		


def show():	
	checkpoint = torch.load('files/result/flip_checkpoint.pth',map_location=torch.device(device))
	state_dict = checkpoint['model_state_dict']
	epoch_count = checkpoint['epoch_count']
	print(epoch_count)
	
	#print(type(state_dict))
	#print(state_dict)
	#print(state_dict[0])
	
	layer1_a1_R = nn.LeakyReLU(negative_slope=1e-4)
	layer1_a1_G = nn.LeakyReLU(negative_slope=1e-4)		
	layer1_a1_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer2_a2_R = nn.LeakyReLU(negative_slope=1e-4)
	layer2_a2_G = nn.LeakyReLU(negative_slope=1e-4)
	layer2_a2_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer3_a3_R = nn.LeakyReLU(negative_slope=1e-4)
	layer3_a3_G = nn.LeakyReLU(negative_slope=1e-4)
	layer3_a3_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer4_a4_R = nn.LeakyReLU(negative_slope=1e-4)
	layer4_a4_G = nn.LeakyReLU(negative_slope=1e-4)
	layer4_a4_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer5_a5_R = nn.LeakyReLU(negative_slope=1e-4)
	layer5_a5_G = nn.LeakyReLU(negative_slope=1e-4)
	layer5_a5_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer6_a6_R = nn.LeakyReLU(negative_slope=1e-4)
	layer6_a6_G = nn.LeakyReLU(negative_slope=1e-4)
	layer6_a6_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer7_a7_R = nn.LeakyReLU(negative_slope=1e-4)
	layer7_a7_G = nn.LeakyReLU(negative_slope=1e-4)
	layer7_a7_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer8_a8_R = nn.LeakyReLU(negative_slope=1e-4)
	layer8_a8_G = nn.LeakyReLU(negative_slope=1e-4)
	layer8_a8_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer9_a9_R = nn.LeakyReLU(negative_slope=1e-4)
	layer9_a9_G = nn.LeakyReLU(negative_slope=1e-4)
	layer9_a9_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer10_a10_R = nn.LeakyReLU(negative_slope=1e-4)
	layer10_a10_G = nn.LeakyReLU(negative_slope=1e-4)
	layer10_a10_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer11_a11_R = nn.LeakyReLU(negative_slope=1e-4)
	layer11_a11_G = nn.LeakyReLU(negative_slope=1e-4)
	layer11_a11_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer12_a12 = nn.Sigmoid()
		
	
	######
	
	input_image = 'files/train_images/65.jpg'
	img = Image.open(input_image)
	img = input_transform(img)
	
	print(img.shape,'orginal_image_shape')
	img = torch.stack([img])
	print(img.shape,'orginal_image_shape')
	
	input_image_r = img[:,:1,:,:]
	input_image_g = img[:,1:2,:,:]
	input_image_b = img[:,2:3,:,:]
	
	layer1_o_r = layer1_a1_R(F.conv2d(input=input_image_r,weight=state_dict['layer1_z1_R.weight'],bias= state_dict['layer1_z1_R.bias'],padding=3))
	layer1_o_g = layer1_a1_G(F.conv2d(input=input_image_g,weight=state_dict['layer1_z1_G.weight'],bias= state_dict['layer1_z1_G.bias'],padding=3))
	layer1_o_b = layer1_a1_B(F.conv2d(input=input_image_b,weight=state_dict['layer1_z1_B.weight'],bias= state_dict['layer1_z1_B.bias'],padding=3))
	layer1_o = torch.stack([layer1_o_r,layer1_o_g,layer1_o_b])
	
	layer2_o_r = layer2_a2_R(F.conv2d(input=layer1_o_r,weight=state_dict['layer2_z2_R.weight'],bias= state_dict['layer2_z2_R.bias'],padding=3))
	layer2_o_g = layer2_a2_G(F.conv2d(input=layer1_o_g,weight=state_dict['layer2_z2_G.weight'],bias= state_dict['layer2_z2_G.bias'],padding=3))
	layer2_o_b = layer2_a2_B(F.conv2d(input=layer1_o_b,weight=state_dict['layer2_z2_B.weight'],bias= state_dict['layer2_z2_B.bias'],padding=3))
	layer2_o = torch.stack([layer2_o_r,layer2_o_g,layer2_o_b])
	
	layer3_o_r = layer3_a3_R(F.conv2d(input=layer2_o_r,weight=state_dict['layer3_z3_R.weight'],bias= state_dict['layer3_z3_R.bias'],padding=3))
	layer3_o_g = layer3_a3_G(F.conv2d(input=layer2_o_g,weight=state_dict['layer3_z3_G.weight'],bias= state_dict['layer3_z3_G.bias'],padding=3))
	layer3_o_b = layer3_a3_B(F.conv2d(input=layer2_o_b,weight=state_dict['layer3_z3_B.weight'],bias= state_dict['layer3_z3_B.bias'],padding=3))
	layer3_o = torch.stack([layer3_o_r,layer3_o_g,layer3_o_b])
	
	layer4_o_r = layer4_a4_R(F.conv2d(input=layer3_o_r,weight=state_dict['layer4_z4_R.weight'],bias= state_dict['layer4_z4_R.bias'],padding=3))
	layer4_o_g = layer4_a4_G(F.conv2d(input=layer3_o_g,weight=state_dict['layer4_z4_G.weight'],bias= state_dict['layer4_z4_G.bias'],padding=3))
	layer4_o_b = layer4_a4_B(F.conv2d(input=layer3_o_b,weight=state_dict['layer4_z4_B.weight'],bias= state_dict['layer4_z4_B.bias'],padding=3))
	layer4_o = torch.stack([layer4_o_r,layer4_o_g,layer4_o_b])
	
	layer5_o_r = layer5_a5_R(F.conv2d(input=layer4_o_r,weight=state_dict['layer5_z5_R.weight'],bias= state_dict['layer5_z5_R.bias'],padding=3))
	layer5_o_g = layer5_a5_G(F.conv2d(input=layer4_o_g,weight=state_dict['layer5_z5_G.weight'],bias= state_dict['layer5_z5_G.bias'],padding=3))
	layer5_o_b = layer5_a5_B(F.conv2d(input=layer4_o_b,weight=state_dict['layer5_z5_B.weight'],bias= state_dict['layer5_z5_B.bias'],padding=3))
	layer5_o = torch.stack([layer5_o_r,layer5_o_g,layer5_o_b])
	
	layer6_o_r = layer6_a6_R(F.conv2d(input=layer5_o_r,weight=state_dict['layer6_z6_R.weight'],bias= state_dict['layer6_z6_R.bias'],padding=3))
	layer6_o_g = layer6_a6_G(F.conv2d(input=layer5_o_g,weight=state_dict['layer6_z6_G.weight'],bias= state_dict['layer6_z6_G.bias'],padding=3))
	layer6_o_b = layer6_a6_B(F.conv2d(input=layer5_o_b,weight=state_dict['layer6_z6_B.weight'],bias= state_dict['layer6_z6_B.bias'],padding=3))
	layer6_o = torch.stack([layer6_o_r,layer6_o_g,layer6_o_b])
	
	print(layer6_o_r.shape) #(batch_size,7,H,W)
	
	layer7_o_r = layer7_a7_R(F.conv2d(input=layer6_o_r,weight=state_dict['layer7_z7_R.weight'],bias= state_dict['layer7_z7_R.bias'],padding=3))
	layer7_o_g = layer7_a7_G(F.conv2d(input=layer6_o_g,weight=state_dict['layer7_z7_G.weight'],bias= state_dict['layer7_z7_G.bias'],padding=3))
	layer7_o_b = layer7_a7_B(F.conv2d(input=layer6_o_b,weight=state_dict['layer7_z7_B.weight'],bias= state_dict['layer7_z7_B.bias'],padding=3))
	layer7_o = torch.stack([layer7_o_r,layer7_o_g,layer7_o_b])
	
	layer8_o_r = layer8_a8_R(F.conv2d(input=layer7_o_r,weight=state_dict['layer8_z8_R.weight'],bias= state_dict['layer8_z8_R.bias'],padding=3))
	layer8_o_g = layer8_a8_G(F.conv2d(input=layer7_o_g,weight=state_dict['layer8_z8_G.weight'],bias= state_dict['layer8_z8_G.bias'],padding=3))
	layer8_o_b = layer8_a8_B(F.conv2d(input=layer7_o_b,weight=state_dict['layer8_z8_B.weight'],bias= state_dict['layer8_z8_B.bias'],padding=3))
	layer8_o = torch.stack([layer8_o_r,layer8_o_g,layer8_o_b])
	
	layer9_o_r = layer9_a9_R(F.conv2d(input=layer8_o_r,weight=state_dict['layer9_z9_R.weight'],bias= state_dict['layer9_z9_R.bias'],padding=3))
	layer9_o_g = layer9_a9_G(F.conv2d(input=layer8_o_g,weight=state_dict['layer9_z9_G.weight'],bias= state_dict['layer9_z9_G.bias'],padding=3))
	layer9_o_b = layer9_a9_B(F.conv2d(input=layer8_o_b,weight=state_dict['layer9_z9_B.weight'],bias= state_dict['layer9_z9_B.bias'],padding=3))
	layer9_o = torch.stack([layer9_o_r,layer9_o_g,layer9_o_b])
	
	layer10_o_r = layer10_a10_R(F.conv2d(input=layer9_o_r,weight=state_dict['layer10_z10_R.weight'],bias= state_dict['layer10_z10_R.bias'],padding=3))
	layer10_o_g = layer10_a10_G(F.conv2d(input=layer9_o_g,weight=state_dict['layer10_z10_G.weight'],bias= state_dict['layer10_z10_G.bias'],padding=3))
	layer10_o_b = layer10_a10_B(F.conv2d(input=layer9_o_b,weight=state_dict['layer10_z10_B.weight'],bias= state_dict['layer10_z10_B.bias'],padding=3))
	layer10_o = torch.stack([layer10_o_r,layer10_o_g,layer10_o_b])
	
	layer11_o_r = layer11_a11_R(F.conv2d(input=layer10_o_r,weight=state_dict['layer11_z11_R.weight'],bias= state_dict['layer11_z11_R.bias'],padding=3))
	layer11_o_g = layer11_a11_G(F.conv2d(input=layer10_o_g,weight=state_dict['layer11_z11_G.weight'],bias= state_dict['layer11_z11_G.bias'],padding=3))
	layer11_o_b = layer11_a11_B(F.conv2d(input=layer10_o_b,weight=state_dict['layer11_z11_B.weight'],bias= state_dict['layer11_z11_B.bias'],padding=3))
	layer11_o = torch.stack([layer11_o_r,layer11_o_g,layer11_o_b])
	
	print(layer11_o_r.shape) #(batch_size,7,H,W)
	
	layer12_o_r = layer12_a12(F.conv2d(input=layer11_o_r,weight=state_dict['layer12_z12_R.weight'],bias= state_dict['layer12_z12_R.bias'],padding=3))
	layer12_o_g = layer12_a12(F.conv2d(input=layer11_o_g,weight=state_dict['layer12_z12_G.weight'],bias= state_dict['layer12_z12_G.bias'],padding=3))
	layer12_o_b = layer12_a12(F.conv2d(input=layer11_o_b,weight=state_dict['layer12_z12_B.weight'],bias= state_dict['layer12_z12_B.bias'],padding=3))
	layer12_o = torch.stack([layer12_o_r,layer12_o_g,layer12_o_b])	
	
	print(layer12_o_r.shape) #(batch_size,1,H,W)
	
	
				
	predicted_o = torch.concatenate((layer12_o_r,layer12_o_g,layer12_o_b),axis = 1)
	#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
	all_layers = torch.stack([layer1_o,layer2_o,layer3_o,layer4_o,layer5_o,layer6_o,layer7_o,layer8_o,layer9_o,layer10_o,layer11_o])
	
	
	#layer1_0 = r(1,7,h,w) , g(1,7,h,w) , b(1,7,h,w)
	for t in range(11):
		for i in range(7):
			plt.subplot(1,7,i+1)
			print(all_layers[t][0][0][i][:,:].shape,'one_channel')
			
			img_r = torch.unsqueeze(all_layers[t][0][0][i][:,:],axis=-1)
			img_g = torch.unsqueeze(all_layers[t][1][0][i][:,:],axis=-1)
			img_b = torch.unsqueeze(all_layers[t][2][0][i][:,:],axis=-1)
			
			image  = torch.concatenate((img_r,img_g,img_b),axis=2)
			print('image:shape',image.shape)
			plt.imshow(image)
			plt.title(f'layer:{t+1} neuron: {i+1}')
		plt.show()
	
	print('epoch_count:',epoch_count)
	
	plt.subplot(1,1,1)
	
	img_r = torch.unsqueeze(layer12_o_r[0][0],axis=-1)
	img_g = torch.unsqueeze(layer12_o_g[0][0],axis=-1)
	img_b = torch.unsqueeze(layer12_o_b[0][0],axis=-1)
	
	img = torch.concatenate((img_r,img_g,img_b),axis=-1)
	plt.imshow(img)
	plt.show()
	
	
	print(checkpoint['loss'])

	
#t = main()
show()








		
		




