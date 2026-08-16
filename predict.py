import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_transform = transforms.Compose([
	transforms.ToTensor()
])

def predict(input_file_path):	
	checkpoint = torch.load('files/result/flip_checkpoint (5).pth',map_location=torch.device(device))
	state_dict = checkpoint['model_state_dict']
	epoch_count = checkpoint['epoch_count']
	print(epoch_count)
	
	layer_activation = nn.LeakyReLU(negative_slope=1e-4)		
	layer12_a12 = nn.Sigmoid()
		
	######
	
	org_img = Image.open(input_file_path)
	org_img_tensor = torch.from_numpy(np.array(org_img))/255
	#print('org_image_shape:',org_img_tensor.shape)
	img = input_transform(org_img)
	
	#print(img.shape,'orginal_image_shape')
	img = torch.stack([img])
	#print(img.shape,'orginal_image_shape')
	
	input_image_r = img[:,:1,:,:]
	input_image_g = img[:,1:2,:,:]
	input_image_b = img[:,2:3,:,:]
	
	layer1_o_r = layer_activation(F.conv2d(input=input_image_r,weight=state_dict['layer1_z1_R.weight'],bias= state_dict['layer1_z1_R.bias'],padding=3))
	layer1_o_g = layer_activation(F.conv2d(input=input_image_g,weight=state_dict['layer1_z1_G.weight'],bias= state_dict['layer1_z1_G.bias'],padding=3))
	layer1_o_b = layer_activation(F.conv2d(input=input_image_b,weight=state_dict['layer1_z1_B.weight'],bias= state_dict['layer1_z1_B.bias'],padding=3))
	layer1_o = torch.stack([layer1_o_r,layer1_o_g,layer1_o_b])
	
	layer2_o_r = layer_activation(F.conv2d(input=layer1_o_r,weight=state_dict['layer2_z2_R.weight'],bias= state_dict['layer2_z2_R.bias'],padding=3))
	layer2_o_g = layer_activation(F.conv2d(input=layer1_o_g,weight=state_dict['layer2_z2_G.weight'],bias= state_dict['layer2_z2_G.bias'],padding=3))
	layer2_o_b = layer_activation(F.conv2d(input=layer1_o_b,weight=state_dict['layer2_z2_B.weight'],bias= state_dict['layer2_z2_B.bias'],padding=3))
	layer2_o = torch.stack([layer2_o_r,layer2_o_g,layer2_o_b])
	
	layer3_o_r = layer_activation(F.conv2d(input=layer2_o_r,weight=state_dict['layer3_z3_R.weight'],bias= state_dict['layer3_z3_R.bias'],padding=3))
	layer3_o_g = layer_activation(F.conv2d(input=layer2_o_g,weight=state_dict['layer3_z3_G.weight'],bias= state_dict['layer3_z3_G.bias'],padding=3))
	layer3_o_b = layer_activation(F.conv2d(input=layer2_o_b,weight=state_dict['layer3_z3_B.weight'],bias= state_dict['layer3_z3_B.bias'],padding=3))
	layer3_o = torch.stack([layer3_o_r,layer3_o_g,layer3_o_b])
	
	layer4_o_r = layer_activation(F.conv2d(input=layer3_o_r,weight=state_dict['layer4_z4_R.weight'],bias= state_dict['layer4_z4_R.bias'],padding=3))
	layer4_o_g = layer_activation(F.conv2d(input=layer3_o_g,weight=state_dict['layer4_z4_G.weight'],bias= state_dict['layer4_z4_G.bias'],padding=3))
	layer4_o_b = layer_activation(F.conv2d(input=layer3_o_b,weight=state_dict['layer4_z4_B.weight'],bias= state_dict['layer4_z4_B.bias'],padding=3))
	layer4_o = torch.stack([layer4_o_r,layer4_o_g,layer4_o_b])
	
	layer5_o_r = layer_activation(F.conv2d(input=layer4_o_r,weight=state_dict['layer5_z5_R.weight'],bias= state_dict['layer5_z5_R.bias'],padding=3))
	layer5_o_g = layer_activation(F.conv2d(input=layer4_o_g,weight=state_dict['layer5_z5_G.weight'],bias= state_dict['layer5_z5_G.bias'],padding=3))
	layer5_o_b = layer_activation(F.conv2d(input=layer4_o_b,weight=state_dict['layer5_z5_B.weight'],bias= state_dict['layer5_z5_B.bias'],padding=3))
	layer5_o = torch.stack([layer5_o_r,layer5_o_g,layer5_o_b])
	
	layer6_o_r = layer_activation(F.conv2d(input=layer5_o_r,weight=state_dict['layer6_z6_R.weight'],bias= state_dict['layer6_z6_R.bias'],padding=3))
	layer6_o_g = layer_activation(F.conv2d(input=layer5_o_g,weight=state_dict['layer6_z6_G.weight'],bias= state_dict['layer6_z6_G.bias'],padding=3))
	layer6_o_b = layer_activation(F.conv2d(input=layer5_o_b,weight=state_dict['layer6_z6_B.weight'],bias= state_dict['layer6_z6_B.bias'],padding=3))
	layer6_o = torch.stack([layer6_o_r,layer6_o_g,layer6_o_b])
	
	#print(layer6_o_r.shape) #(batch_size,7,H,W)
	
	layer7_o_r = layer_activation(F.conv2d(input=layer6_o_r,weight=state_dict['layer7_z7_R.weight'],bias= state_dict['layer7_z7_R.bias'],padding=3))
	layer7_o_g = layer_activation(F.conv2d(input=layer6_o_g,weight=state_dict['layer7_z7_G.weight'],bias= state_dict['layer7_z7_G.bias'],padding=3))
	layer7_o_b = layer_activation(F.conv2d(input=layer6_o_b,weight=state_dict['layer7_z7_B.weight'],bias= state_dict['layer7_z7_B.bias'],padding=3))
	layer7_o = torch.stack([layer7_o_r,layer7_o_g,layer7_o_b])
	
	layer8_o_r = layer_activation(F.conv2d(input=layer7_o_r,weight=state_dict['layer8_z8_R.weight'],bias= state_dict['layer8_z8_R.bias'],padding=3))
	layer8_o_g = layer_activation(F.conv2d(input=layer7_o_g,weight=state_dict['layer8_z8_G.weight'],bias= state_dict['layer8_z8_G.bias'],padding=3))
	layer8_o_b = layer_activation(F.conv2d(input=layer7_o_b,weight=state_dict['layer8_z8_B.weight'],bias= state_dict['layer8_z8_B.bias'],padding=3))
	layer8_o = torch.stack([layer8_o_r,layer8_o_g,layer8_o_b])
	
	layer9_o_r = layer_activation(F.conv2d(input=layer8_o_r,weight=state_dict['layer9_z9_R.weight'],bias= state_dict['layer9_z9_R.bias'],padding=3))
	layer9_o_g = layer_activation(F.conv2d(input=layer8_o_g,weight=state_dict['layer9_z9_G.weight'],bias= state_dict['layer9_z9_G.bias'],padding=3))
	layer9_o_b = layer_activation(F.conv2d(input=layer8_o_b,weight=state_dict['layer9_z9_B.weight'],bias= state_dict['layer9_z9_B.bias'],padding=3))
	layer9_o = torch.stack([layer9_o_r,layer9_o_g,layer9_o_b])
	
	layer10_o_r = layer_activation(F.conv2d(input=layer9_o_r,weight=state_dict['layer10_z10_R.weight'],bias= state_dict['layer10_z10_R.bias'],padding=3))
	layer10_o_g = layer_activation(F.conv2d(input=layer9_o_g,weight=state_dict['layer10_z10_G.weight'],bias= state_dict['layer10_z10_G.bias'],padding=3))
	layer10_o_b = layer_activation(F.conv2d(input=layer9_o_b,weight=state_dict['layer10_z10_B.weight'],bias= state_dict['layer10_z10_B.bias'],padding=3))
	layer10_o = torch.stack([layer10_o_r,layer10_o_g,layer10_o_b])
	
	layer11_o_r = layer_activation(F.conv2d(input=layer10_o_r,weight=state_dict['layer11_z11_R.weight'],bias= state_dict['layer11_z11_R.bias'],padding=3))
	layer11_o_g = layer_activation(F.conv2d(input=layer10_o_g,weight=state_dict['layer11_z11_G.weight'],bias= state_dict['layer11_z11_G.bias'],padding=3))
	layer11_o_b = layer_activation(F.conv2d(input=layer10_o_b,weight=state_dict['layer11_z11_B.weight'],bias= state_dict['layer11_z11_B.bias'],padding=3))
	layer11_o = torch.stack([layer11_o_r,layer11_o_g,layer11_o_b])
	
	#print(layer11_o_r.shape) #(batch_size,7,H,W)
	
	layer12_o_r = layer12_a12(F.conv2d(input=layer11_o_r,weight=state_dict['layer12_z12_R.weight'],bias= state_dict['layer12_z12_R.bias'],padding=3))
	layer12_o_g = layer12_a12(F.conv2d(input=layer11_o_g,weight=state_dict['layer12_z12_G.weight'],bias= state_dict['layer12_z12_G.bias'],padding=3))
	layer12_o_b = layer12_a12(F.conv2d(input=layer11_o_b,weight=state_dict['layer12_z12_B.weight'],bias= state_dict['layer12_z12_B.bias'],padding=3))
	layer12_o = torch.stack([layer12_o_r,layer12_o_g,layer12_o_b])	
	
	#print(layer12_o_r.shape) #(batch_size,1,H,W)
	
	predicted_o = torch.concatenate((layer12_o_r,layer12_o_g,layer12_o_b),axis = 1)
	#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
	all_layers = torch.stack([layer1_o,layer2_o,layer3_o,layer4_o,layer5_o,layer6_o,layer7_o,layer8_o,layer9_o,layer10_o,layer11_o])
	
	
	#layer1_0 = r(1,7,h,w) , g(1,7,h,w) , b(1,7,h,w)
	for t in range(11):
		for i in range(7):
			#plt.subplot(1,7,i+1)
			#print(all_layers[t][0][0][i][:,:].shape,'one_channel')
			
			img_r = torch.unsqueeze(all_layers[t][0][0][i][:,:],axis=-1)
			img_g = torch.unsqueeze(all_layers[t][1][0][i][:,:],axis=-1)
			img_b = torch.unsqueeze(all_layers[t][2][0][i][:,:],axis=-1)
			
			image  = torch.concatenate((img_r,img_g,img_b),axis=2)
			#print('image:shape',image.shape)
			#plt.imshow(image)
			#plt.title(f'layer:{t+1} neuron: {i+1}')
		#plt.show()
	
	print('epoch_count:',epoch_count)
	img_r = torch.unsqueeze(layer12_o_r[0][0],axis=-1)
	img_g = torch.unsqueeze(layer12_o_g[0][0],axis=-1)
	img_b = torch.unsqueeze(layer12_o_b[0][0],axis=-1)
	img = torch.concatenate((img_r,img_g,img_b),axis=-1)

	plt.subplot(1,2,2)
	plt.title('predicted_flipped_image')
	plt.imshow(img)
	
	plt.subplot(1,2,1)
	plt.title('Original_Image')
	plt.imshow(org_img_tensor)
	plt.show()
	
	print(checkpoint['loss'])

#65,870,6589,20589,43211
predict('files/train_images/43211.jpg')
