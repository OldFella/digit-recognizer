import torch
import torchvision
import torchvision.transforms as transforms

import os

train_transforms = transforms.Compose([
	transforms.CenterCrop(28),
	# transforms.RandomVerticalFlip(0.5),
	# transforms.RandomHorizontalFlip(0.5),
	# transforms.RandomRotation(30),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

validation_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


train_set = torchvision.datasets.ImageFolder(
	'data/train/',
	transform = train_transforms)

validation_set = torchvision.datasets.ImageFolder(
	'data/validation/',
	transform = validation_transforms)

train_loader = torch.utils.data.DataLoader(
	train_set,
	batch_size = 86,
	num_workers = 2,
	shuffle = True)


validation_loader = torch.utils.data.DataLoader(
	validation_set,
	batch_size = 86,
	num_workers = 2,
	shuffle = True)


def create_loader(folder, transforms, batch_size = 4):
	dataset = torchvision.datasets.ImageFolder(
		root = 'data/' + folder,
		transform = transforms)
	
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 2)

	return loader