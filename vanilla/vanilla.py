import numpy as np
from random import randint
from numpy import array
from numpy import argmax
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.regularizers import Regularizer
from keras import optimizers
def generate_sequence(length, n_features):
	return [randint(0, n_features-1) for _ in range(length)]

def one_hot_encode(sequence, n_features):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_features)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)
def one_hot_decode(encoded_seq):
	#argmax gives largest index when working on array
	return [argmax(vector) for vector in encoded_seq]
def generate_example(length, n_features, out_index):
	sequence = generate_sequence(length, n_features)
	encoded = one_hot_encode(sequence, n_features)
	
	X = encoded.reshape((1, length, n_features))
	y = encoded[out_index].reshape(1, n_features)
	return X, y
length = 10
n_features = 15
out_index = 6
model = Sequential()
#, W_regularizer=keras.regularizers.l2(0.000001)
model.add(LSTM(100, input_shape=[length, n_features], dropout_W=0.5))

model.add(Dense(n_features, activation ='softmax'))
sgd = optimizers.SGD(lr=0.5, decay=1e-4, momentum=0.90, nesterov=True, clipvalue=0.005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metric=['accuracy'])
model.summary()

for i in range(20000):
	X, y = generate_example(length, n_features, out_index)
	model.fit(X, y, nb_epoch=1, verbose=2, batch_size=1)

correct = 0
for i in range(100):
	X, y = generate_example(length, n_features, out_index)
	yhat = model.predict(X)
	if one_hot_decode(yhat) == one_hot_decode(y):
		correct += 1
print('Accuracy: %f' % ((correct/100)*100.0))

X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence: %s' % [one_hot_decode(x) for x in X])
print('Expected: %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))
