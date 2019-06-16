import os
from sklearn.externals import joblib
import numpy as np
import keras


def customization_generator(path, batch_size):
	while 1:
		cnt = 0
		X1 = []
		X2 = []
		Y = []
		for filename in os.listdir(path):
			filepath = os.path.join(path, filename)
			x1, x2, y = joblib.load(filepath)
			X1.append(x1)
			X2.append(x2)
			Y.append(y)
			cnt += 1
			if cnt == batch_size:
				cnt = 0
				yield [np.array(X1), np.array(X2)], np.array(Y)
				X1 = []
				X2 = []
				Y = []
