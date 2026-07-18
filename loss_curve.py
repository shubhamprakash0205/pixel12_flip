import os
import torch
import matplotlib.pyplot as plt


checkpoint_path = f'files/result/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
	files = os.listdir(checkpoint_path)
	files = [os.path.join(checkpoint_path,file) for file in files if os.path.isfile(os.path.join(checkpoint_path,file))]
	epoch_dict = []
	loss_dict = []
	lr_dict = []
	for file in files:
		checkpoint = torch.load(file,map_location=device)
		epoch_count = checkpoint['epoch_count']
		loss = checkpoint['loss']
		lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
		lr_dict.append(lr)
		epoch_dict.append(epoch_count)
		loss_dict.append(loss.item())
	lr_dict = torch.tensor(lr_dict)
	epoch_dict = torch.tensor(epoch_dict)
	loss_dict = torch.tensor(loss_dict)
	sorted_arg = torch.argsort(epoch_dict)
	epoch_dict = epoch_dict[sorted_arg]
	loss_dict = loss_dict[sorted_arg]
	lr_dict = lr_dict[sorted_arg]
	print(epoch_dict)
	print(loss_dict)
	print(lr_dict)	
	plt.plot(epoch_dict,lr_dict,marker='o',linestyle='-',label='Loss Curve')
	plt.xlabel('Epoch_count')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('loss_curve.png')
	plt.show()
		
	
	
main()	
	
	
	
	
	
	
	
	 


