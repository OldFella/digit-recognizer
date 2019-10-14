import pandas as pd
import torch
import torchvision.transforms as transforms
from DigitNet import DigitNet
import os
from load_data import validation_transforms
import PIL.Image as Image

submission = pd.read_csv('sample_submission.csv')
# print(submisssion)
model = torch.load('15epochs_0.0007learingrate_32batchsize.pt')
model.eval()

# trans = transforms.ToTensor()

for sample in range(0, len(os.listdir('data/test/'))):
	im = Image.open('data/test/' + str(sample) + '.png')
	im = im.convert('RGB')
	im = validation_transforms(im)
	im = im[None,:,:,:]
	output = model(im)
	pred = output.argmax(dim = 1, keepdim = True)
	# print(pred.item())
	submission['Label'][sample] = pred.item()
# submission.drop(submission.columns[0], axis = 1, inplace = True)
submission.to_csv('sample_submission.csv')
