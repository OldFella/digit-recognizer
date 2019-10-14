import os
import torch

from torch import nn
# from _config import MODEL_PATH, SPLIT_DATA_PATH
from train import train
from test import test
from DigitNet import DigitNet
# from prepare_data import create_loader, test_transforms, test_loader, validation_loader
# from visualize import plot_random_batch, loss_to_epochs, plot_two_graphs
from load_data import validation_loader
from datetime import datetime
import torchvision.models as models


# init model
DIGIT_NET = DigitNet()
# DIGIT_NET = models.DenseNet()

# set hyperparameters
TRAINING_EPOCHS = 15
NUMBER_OF_TESTS = 1
LEARNING_RATE = 0.0007
BATCH_SIZE = 32

# start timer
start = datetime.now()

# train
model = train(
	model = DIGIT_NET,
	criterion = nn.CrossEntropyLoss(),
	training_epochs = TRAINING_EPOCHS,
	learning_rate = LEARNING_RATE,
	batch_size = BATCH_SIZE)

#test
for i in range(NUMBER_OF_TESTS):
 	test(validation_loader, model)


print("\nOverall training and testing time: " + str(datetime.now() - start))

# plot loss over epochs
# loss_to_epochs(graph[0], graph[1], graph[2])
# plot_two_graphs(graph[0], graph[1])

