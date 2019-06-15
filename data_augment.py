from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from skimage import io, transform


sourse_path = r'dataset'
dest_path = r'dataset_aug'

# width, height and channel
# If you want to process a RGB image, remember to change the channel to 3
w = 105
h = 105
c = 1

category_num = 394			# The number of image category
generate_num = 10			# The number of new images you want to generate per image
							# Total number of images generated = generate_num * number of images in folder


def read_img(path):
	'''
	Read all image in the directory.
	:param path: Folder path.
	:return:A numpy array conversion from images.
			For RGB image, the shape is (number of image, width, height, 3).
	'''
	imgs = []
	for filename in os.listdir(path):
		print(filename)
		file_path = os.path.join(path, filename)
		img = io.imread(file_path)
		x = transform.resize(img, (w, h, c), mode='constant')
		imgs.append(x)
	return np.asarray(imgs, np.float32)

if __name__ == '__main__':
	new_image = ImageDataGenerator(rotation_range=40,
								   width_shift_range=0.1,
								   height_shift_range=0.1,
								   shear_range=0.1,
								   zoom_range=0.2,
								   fill_mode='nearest')  # remember to change fill_mode as needed

	for j in range(category_num):
		data_path = os.path.join(sourse_path, str(j))
		print(data_path)
		data = read_img(data_path)
		save_path = os.path.join(dest_path, str(j))
		img_num = len(os.listdir(data_path))
		prefix = 0
		for x in range(img_num):
			base_data = data[x].reshape((1,) + data[x].shape)
			seed = 1
			i = 0
			for batch in new_image.flow(base_data, batch_size=1, seed=seed,
										save_to_dir=save_path, save_prefix=prefix, save_format='jpg'):
				i += 1
				prefix += 1
				if i >= generate_num:
					break

		k = 0
		for filename in os.listdir(save_path):
			os.rename(os.path.join(save_path, filename), os.path.join(save_path, str(k) + '.jpg'))
			k += 1
