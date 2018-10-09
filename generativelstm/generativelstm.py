from keras.models import Sequential
from keras.layers import LSTM, Dense
from numpy import array
from randomrect import get_samples, plot_rectangle

model = Sequential()
model.add(LSTM(10, input_shape=(1, 2)))
model.add(Dense(2, activation='linear'))
model.compile(loss='mae', optimizer='adam', metrics=['acc'])
model.summary()

#this isn't useful yet

histories = list()

for i in range(25000):
    X, y = get_samples()
    history = model.fit(X, y, epochs=1, verbose=2, shuffle=False)
    histories.append(history.history)
print(histories)

def generate_rectangle(model):
    rect = list()
    last = array([0.0, 0.0]).reshape((1,1,2))
    rect.append([[y for y in x] for x in last[0]][0])

    for i in range(3):
        yhat = model.predict(last, verbose = 0)

        last = yhat.reshape((1, 1, 2))

        rect.append([[y for y in x] for x in last[0]][0])
    return rect

rect = generate_rectangle(model)
plot_rectangle(rect)


