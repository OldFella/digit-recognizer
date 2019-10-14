import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from PIL import Image
import os
# print(os.listdir())

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_label = train['label']
train_data = train.drop(labels = ['label'], axis = 1)

# print(train_data)

del train

# train_data = train_data/255.0
# train_data = train_data.values.reshape(-1,28,28,1)
# test = test.values.reshape(-1,28,28,1)

# print(train_data.shape)
def csv_to_array(csv, sizeX):
	s = csv.shape
	x_bounds = s[1]
	y_bounds = s[0]
	# print(x_bounds, y_bounds)
	result = []
	for row in range(0, y_bounds):
		image = []
		pixel_row = []
		for column in range(0,x_bounds):
			pixel_row.append(csv['pixel' + str(column)][row])
			if column % sizeX == 27:
				image.append(np.asarray(pixel_row))
				# print(len(pixel_row))
				pixel_row = []
		# print(row)
		# print(result[row])
		if row % 1000 == 0:
			print(row)
		result.append(np.asarray(image))

	return result
train_data = csv_to_array(train_data, 28)
test = csv_to_array(test,28)
# print(test[0])

train_dir = '/home/gwent/projects/kaggle_comp/digit-recognizer/data/train/'
test_dir = 'data/test/'
for image in range(0, len(train_data)):

	im = Image.fromarray(np.uint8(train_data[image]))
	# print(train_label[image])
	im.save(train_dir + str(train_label[image])+ '/' + str(image) + '.png', 'png')


for image in range(0, len(test)):

	im = Image.fromarray(np.uint8(test[image]))
	# print(train_label[image])
	im.save(test_dir + '/' + str(image) + '.png', 'png')
# # train_label = to_categorical(train_label, num_classes = 10)

# # print(train_data[0])

# train_set = torch.utils.data.DataLoader(
# 	train_data,
# 	batch_size = 86,
# 	num_workers = 2)

# mnist = torchvision.datasets.FashionMNIST('', download = True)