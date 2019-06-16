import os
import numpy as np
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

root_path = r'Omniglot Processed/train image'
data_path = r'dataset_for_gen'


def load_data(image_num, data_number, seq_len):
	traingen = ImageDataGenerator(rescale=1. / 255)
	# 生成图片对
	print('loading data.....')
	frame_num = image_num
	train_data1 = []
	train_data2 = []
	train_lab = []
	count = 0
	while count < data_number:
		count = count + 1
		save_path = os.path.join(data_path, str(count) + '.pkl')
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
			train_data1 = np.squeeze(image1)
			train_data2 = np.squeeze(image2)
			train_lab = np.array(0)
			joblib.dump((train_data1, train_data2, train_lab), save_path, compress=0, protocol=4)
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
			train_data1 = np.squeeze(image1)
			train_data2 = np.squeeze(image2)
			train_lab = np.array(1)
			joblib.dump((train_data1, train_data2, train_lab), save_path, compress=0, protocol=4)
	print("loading finish!")
	return [np.array(train_data1), np.array(train_data2), np.array(train_lab)]


if __name__ == '__main__':
	load_data(20, 100, 964)
