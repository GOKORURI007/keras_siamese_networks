import os
import shutil

dataset_path_tr = r'Omniglot Dataset/images_background'
dataset_path_te = r'Omniglot Dataset/images_evaluation'


def create_dir(path, num_of_dir):
	for i in range(num_of_dir):
		os.mkdir(os.path.join(path, str(i)))


if __name__ == '__main__':

	os.mkdir('Omniglot Processed')
	os.mkdir(r'Omniglot Processed\train image')
	os.mkdir(r'Omniglot Processed\test image')
	train_path = r'Omniglot Processed\train image'
	test_path = r'Omniglot Processed\test image'

	type = os.listdir(dataset_path_tr)
	typenum = len(type)
	characternum = 0
	dest_dirname = 0

	for i in range(typenum):
		type_path = os.path.join(dataset_path_tr, type[i])
		character = os.listdir(type_path)
		characternum += len(character)
	create_dir(train_path, characternum)

	for i in range(typenum):
		print('process the ' + str(i) + ' th character...')
		type_path = os.path.join(dataset_path_tr, type[i])
		character = os.listdir(type_path)
		for j in range(len(character)):
			character_path = os.path.join(type_path, character[j])
			image = os.listdir(character_path)
			for x in range(len(image)):
				image_path = os.path.join(character_path, image[x])
				shutil.copyfile(image_path, os.path.join(train_path, str(dest_dirname), str(x) + '.jpg'))
			dest_dirname += 1

	type = os.listdir(dataset_path_te)
	typenum = len(type)
	characternum = 0
	dest_dirname = 0

	for i in range(typenum):
		type_path = os.path.join(dataset_path_te, type[i])
		character = os.listdir(type_path)
		characternum += len(character)
	create_dir(test_path, characternum)

	for i in range(typenum):
		print('process the ' + str(i) + ' th character...')
		type_path = os.path.join(dataset_path_te, type[i])
		character = os.listdir(type_path)
		for j in range(len(character)):
			character_path = os.path.join(type_path, character[j])
			image = os.listdir(character_path)
			for x in range(len(image)):
				image_path = os.path.join(character_path, image[x])
				shutil.copyfile(image_path, os.path.join(test_path, str(dest_dirname), str(x) + '.jpg'))
			dest_dirname += 1
