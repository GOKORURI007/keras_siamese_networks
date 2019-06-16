from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, concatenate, Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name,
			   kernel_initializer='uniform')(x)
	x = BatchNormalization(axis=3, name=bn_name)(x)
	x = Activation('relu')(x)
	return x


def AlexNet(w, h, c):
	'''AlexNet with out Dense layers'''
	input = Input(shape=(w, h, c))
	x = Conv2d_BN(input, 96, (11, 11), strides=(4, 4), padding='valid')
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
	x = Conv2d_BN(x, 256, (5, 5), strides=(1, 1), padding='same')
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
	x = Conv2d_BN(x, 384, (3, 3), strides=(1, 1), padding='same')
	x = Conv2d_BN(x, 384, (3, 3), strides=(1, 1), padding='same')
	x = Conv2d_BN(x, 256, (3, 3), strides=(1, 1), padding='same')
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
	output = Flatten()(x)
	# output = Dropout(0.25)
	model = Model(input, output, name='AlexNetBase')
	return model
