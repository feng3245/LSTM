from numpy import zeros, array
from random import randint
from random import random
from matplotlib import pyplot

def next_frame(last_step, last_frame, column):
    lower = max(0, last_step - 1)
    upper = min(last_frame.shape[0]-1, last_step+1)

    step = randint(lower, upper)

    frame = last_frame.copy()

    frame[step, column] = 1
    return frame, step

def build_frames(size):
    frames = list()

    frame = zeros((size,size))
    step = randint(0, size-1)

    right = 1 if random() < 0.5 else 0
    col = 0 if right else size-1
    frame[step, col] = 1
    frames.append(frame)

    for i in range(1, size):
        col = i if right else size-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right

def generate_examples(size, n_patterns):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(size)
        X.append(frames)
        y.append(right)

    X = array(X).reshape(n_patterns, size, size, size, 1)
    y = array(y).reshape(n_patterns, 1)
    return X, y





