import matplotlib.pyplot as plt
from lib import *

def show_needle_map(correspondences, c='r', ax=None):
    global plt
    if ax is not None:
        plt = ax
    for x1, y1, x2, y2 in correspondences:
        plt.plot([x1, x2], [y1, y2], c=c, linewidth=1)
    plt.scatter(correspondences[:, 0], correspondences[:, 1], marker='o', s=5, c=c)

def show_image(image, ax):
    height, width = image.shape[0], image.shape[1]
    ax.imshow(image, alpha=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()

def show_line(line, width, height, ax):

    l1 = line_from_points(0, 0, width, 0)
    l2 = line_from_points(0, height, width, height)

    x1, y1 = lines_intersection(line, l1)
    x2, y2 = lines_intersection(line, l2)

    ax.plot([x1, x2], [y1, y2])


def show_epipolar_lines(correspondences, F, ax1, ax2, width, height, each=100):
    N = len(correspondences)
    
    for i in range(0, N, each):
        u1, v1, u2, v2 = correspondences[i]
        m1 = np.array([u1, v1, 1])
        m2 = np.array([u2, v2, 1])

        l1 = F.T @ m2
        l2 = F @ m1

        ax1.scatter(u1, v1)
        ax2.scatter(u2, v2)

        show_line(l1, width, height, ax1)
        show_line(l2, width, height, ax2)