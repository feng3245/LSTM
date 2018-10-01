from math import ceil, log10
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from generatesample import generate_data, invert

n_terms = 3

largest = 10

alphabet = [str(x) for x in range(10)] + ['+', ' ']

n_chars = len(alphabet)

n_in_seq_length = int(n_terms * ceil(log10(largest+1)) + n_terms - 1)

n_out_seq_length = int(ceil(log10(n_terms * (largest+1))))

model = Sequential()
model.add(LSTM(75, input_shape=(n_in_seq_length, n_chars)))
model.add(RepeatVector(n_out_seq_length))
model.add(LSTM(50, return_sequences = True))
model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

X, y = generate_data(75000, n_terms, largest, alphabet)
model.fit(X, y, epochs=1, batch_size=32)

X, y = generate_data(100, n_terms, largest, alphabet)
loss, acc = model.evaluate(X, y, verbose = 0)
print('Loss %f, Accuracy: %f' % (loss, acc*100))

for _ in range(10):
    X, y = generate_data(1, n_terms, largest, alphabet)

    yhat = model.predict(X, verbose=0)

    in_seq = invert(X[0], alphabet)
    out_seq = invert(y[0], alphabet)
    predicted = invert(yhat[0], alphabet)
    print('%s = %s (expect %s)' % (in_seq, predicted, out_seq))

