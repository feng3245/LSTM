from random import random
from numpy import array
from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def random_rectangle():
    width, height = random(), random()
    points = list()

    points.append([0.0, 0.0])

    points.append([width, 0.0])

    points.append([width, height])

    points.append([0.0, height])
    return points

def plot_rectangle(rect):
    rect.append(rect[0])

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(rect, codes)
    axis = pyplot.gca()
    patch = PathPatch(path)

    axis.add_patch(patch)
    axis.set_xlim(-0.1, 1.1)
    axis.set_ylim(-0.1, 1.1)
    pyplot.show()


def get_samples():
    rect = random_rectangle()
    X, y = list(), list()

    for i in range(1, len(rect)):
        X.append(rect[i-1])
        y.append(rect[i])
    X, y = array(X), array(y)
    X = X.reshape((X.shape[0], 1, 2))
    return X, y
    

