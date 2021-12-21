import matplotlib.pyplot as plt
import numpy as np


def create_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-1, 5)

    return fig, ax


def show_camera_3d(ax, Rt, text=''):
    R, t = Rt[:, :3], Rt[:, 3]

    ax.quiver(*t, *(R.T @ ([1, 0, 0])), color='b', arrow_length_ratio=0.2)
    ax.quiver(*t, *(R.T @ ([0, 1, 0])), color='g', arrow_length_ratio=0.2)
    ax.quiver(*t, *(R.T @ ([0, 0, 1])), color='r', arrow_length_ratio=0.2)
    
    ax.text(*(t + -1e-1), text)
    ax.scatter(*t, c='k')

fig, ax = create_3d_plot()

Rts = np.load('Rt.npy')
X = np.load('X.npy')

for i, Rt in enumerate(Rts):
    show_camera_3d(ax, Rt, i + 1)

ax.scatter(X[0], X[1], X[2], marker='.', s=3)
plt.show()