import os
import numpy as np
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

root_path = r'Omniglot Processed/train image'
save_file = r'dataset_tr.pickle'


def load_data(image_num, data_number, seq_len):
	'''
	generate image pairs and label for train then save them as a tuple of numpy arrays.
	:param data_number: number of image pairs you want to generate.
	:param seq_len:number of image categories.
	:param image_num:The number of images in each category.
	:return:A tuple of numpy arrays, (images 1, images 2, labels).
	'''
	traingen = ImageDataGenerator(rescale=1. / 255)	 # only normalize
	print('loading data.....')
	frame_num = image_num
	train_data1 = []
	train_data2 = []
	train_lab = []
	count = 0
	while count < data_number:
		count = count + 1
		print("Generating the ", count, "th image pair")
		pos_neg = np.random.randint(0, 2)
		if pos_neg == 0:
			seed1 = np.random.randint(0, seq_len)
			seed2 = np.random.randint(0, seq_len)
			while seed1 == seed2:
				seed1 = np.random.randint(0, seq_len)
				seed2 = np.random.randint(0, seq_len)
			frame1 = np.random.randint(0, frame_num)
			frame2 = np.random.randint(0, frame_num)
			path1 = os.path.join(root_path, str(seed1), str(frame1) + '.jpg')
			path2 = os.path.join(root_path, str(seed2), str(frame2) + '.jpg')
			image1 = img_to_array(load_img(path1))
			image1 = image1.reshape((1,) + image1.shape)
			for new_img in traingen.flow(image1, batch_size=1, shuffle=False):
				image1 = new_img
				break
			image2 = img_to_array(load_img(path2))
			image2 = image2.reshape((1,) + image2.shape)
			for new_img in traingen.flow(image2, batch_size=1, shuffle=False):
				image2 = new_img
				break
			train_data1.append(np.squeeze(image1))
			train_data2.append(np.squeeze(image2))
			train_lab.append(np.array(0))

		else:
			seed = np.random.randint(0, seq_len)
			frame1 = np.random.randint(0, frame_num)
			frame2 = np.random.randint(0, frame_num)
			path1 = os.path.join(root_path, str(seed), str(frame1) + '.jpg')
			path2 = os.path.join(root_path, str(seed), str(frame2) + '.jpg')
			image1 = img_to_array(load_img(path1))
			image1 = image1.reshape((1,) + image1.shape)
			for new_img in traingen.flow(image1, batch_size=1, shuffle=False):
				image1 = new_img
				break
			image2 = img_to_array(load_img(path2))
			image2 = image2.reshape((1,) + image2.shape)
			for new_img in traingen.flow(image2, batch_size=1, shuffle=False):
				image2 = new_img
				break
			train_data1.append(np.squeeze(image1))
			train_data2.append(np.squeeze(image2))
			train_lab.append(np.array(1))
	print("loading finish!")
	return np.array(train_data1), np.array(train_data2), np.array(train_lab)


if __name__ == '__main__':
	train_data1, train_data2, train_lab = load_data(20, 100, 964)
	print("start dumping!")
	# higher protocol, faster speed.
	# higher compress, smaller file size, but also slower read and write times.
	joblib.dump((train_data1, train_data2, train_lab), save_file, protocol=4, compress=0)
	print("dumping finish!")