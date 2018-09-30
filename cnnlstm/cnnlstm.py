from keras.models import Sequential
from keras.layers import Conv2D, TimeDistributed, Flatten
from keras.layers import MaxPooling2D, LSTM, Dense
from movingsqrVidPred import generate_examples

size = 50
model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2,2), activation='relu'), input_shape=(None, size, size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

X, y = generate_examples(size, 5000)
model.fit(X, y, batch_size=32, epochs=1)

X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0] == 1 else "Left"
predicted = "Right" if yhat[0] == 1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))
