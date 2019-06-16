import numpy as np
from sklearn.externals import joblib
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.layers import  Lambda
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


def euclidean_distance(vects):
	'''计算欧式距离'''
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	'''欧氏距离输出向量维度'''
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	'''Contrastive loss(对比损失函数) from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	pred = y_pred.ravel() < 0.5
	return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_network():
	'''Puts your base network here'''
	base_model = InceptionV3(input_shape=(105,105,3),include_top=False, weights='imagenet', pooling='avg')
	return base_model


if __name__ == '__main__':
	# dataset loading
	train_data1, train_data2, train_lab = joblib.load('dataset_tr.pickle')
	test_data1, test_data2, test_lab = joblib.load('dataset_te.pickle')
	# saimese network definition
	K.clear_session()
	base_network = create_base_network()
	input_a = Input(shape=(105, 105, 3))
	input_b = Input(shape=(105, 105, 3))
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	distance = Lambda(euclidean_distance,
					  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	model = Model([input_a, input_b], distance)

	# model.load_weights('InceptionV3_Weight.h5')
	# train
	# 返回函数设置，学习率调整
	rms = RMSprop(lr=0.0001)
	reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
	filepath = 'InceptionV3_BestWeight.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callback_list = [checkpoint, reduce_lr]

	# 编译模型并训练
	model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
	model.fit([train_data1, train_data2],
			  train_lab,
			  callbacks=callback_list,
			  batch_size=64, # if you get a memory warning, reduce it.
			  epochs=50)
	model.save_weights('InceptionV3_Weight.h5')

	# compute final accuracy on training and test sets
	y_pred = model.predict([train_data1, train_data2])
	tr_acc = compute_accuracy(train_lab, y_pred)
	y_pred = model.predict([test_data1, test_data2])
	te_acc = compute_accuracy(test_lab, y_pred)

	print('* Accuracy on training set: %0.2f%%' % float(100 * tr_acc))
	print('* Accuracy on test set: %0.2f%%' % float(100 * te_acc))
