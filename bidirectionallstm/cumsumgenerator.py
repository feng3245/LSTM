from random import random
from numpy import array
from numpy import cumsum

def get_sequence(n_timesteps):
    X = array([random() for _ in range(n_timesteps)])
    limit = n_timesteps/4.0

    y = array([0 if x < limit else 1 for x in cumsum(X)])
    return X, y


def get_sequences(n_sequences, n_timesteps):
    seqX, seqY = list(), list()

    for _ in range(n_sequences):
        X, y = get_sequence(n_timesteps)
        seqX.append(X)
        seqY.append(y)
    seqX = array(seqX).reshape(n_sequences, n_timesteps, 1)
    seqY = array(seqY).reshape(n_sequences, n_timesteps, 1)
    return seqX, seqY
