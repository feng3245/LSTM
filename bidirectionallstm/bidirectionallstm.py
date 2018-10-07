from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from cumsumgenerator import get_sequences
from numpy import array_equal
n_timesteps = 10
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences = True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

X, y = get_sequences(50000, n_timesteps)
model.fit(X, y, epochs=1, batch_size=10)

X, y = get_sequences(100, n_timesteps)
loss, acc = model.evaluate(X, y, verbose = 0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100))

for _ in range(10):
    X, y = get_sequences(1, n_timesteps)
    yhat = model.predict_classes(X, verbose=0)
    exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)
    print('y=%s, yhat=%s, correct=%s' % (exp, pred, array_equal(exp, pred)))
