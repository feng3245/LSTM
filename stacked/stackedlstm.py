from keras.models import Sequential
from keras.layers import LSTM, Dense
from randdampedsinewave import generate_examples
from matplotlib import pyplot
length = 50
output = 5

model = Sequential()
model.add(LSTM(30, return_sequences = True, input_shape=(length, 1)))
model.add(LSTM(30))
model.add(Dense(output))
model.compile(loss='mae', optimizer='adam')
model.summary()

X, y = generate_examples(length, 10000, output)
model.fit(X, y, batch_size=10, nb_epoch=1)

X, y = generate_examples(length, 1000, output)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)

X, y = generate_examples(length, 1, output)
yhat = model.predict(X, verbose = 0)
pyplot.plot(y[0], label='y')
pyplot.plot(yhat[0], label='yhat')
pyplot.legend()
pyplot.show()
