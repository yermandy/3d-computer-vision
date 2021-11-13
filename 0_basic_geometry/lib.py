import numpy as np
import matplotlib.pyplot as plt
import math
import random

def lines_intersection(l1, l2):
    x = np.cross(l1, l2)
    x = [x[0] / x[2], x[1] / x[2]]
    return x

def line_from_points(x1, y1, x2, y2):
    return np.cross([x1, y1, 1], [x2, y2, 1])
