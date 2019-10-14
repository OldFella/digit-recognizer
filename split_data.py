import os
import random
import shutil

train_dir = 'data/train/'
validation_dir = 'data/validation/'

for x in range(0,10):
	files = os.listdir(train_dir + str(x))
	for file in files:
		random_number = random.random()
		if random_number <= 0.1:
			shutil.move(train_dir + str(x) +'/'+ file, validation_dir + str(x)+ '/' + file)