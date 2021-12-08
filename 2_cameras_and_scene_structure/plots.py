import matplotlib.pyplot as plt
from lib import *

def show_needle_map(correspondences, c='r', ax=None):
    global plt
    if ax is not None:
        plt = ax
    for x1, y1, x2, y2 in correspondences:
        plt.plot([x1, x2], [y1, y2], c=c, linewidth=0.75, alpha=0.75)
    plt.scatter(correspondences[:, 0], correspondences[:, 1], marker='o', s=2.5, c=c)

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


def show_epipolar_lines(correspondences, F, ax1, ax2, width, height, n_lines=10):
    N = len(correspondences)

    indices = np.random.choice(range(N), size=n_lines, replace=False)
    correspondences = correspondences[indices]
    
    for u1, v1, u2, v2 in correspondences:
        m1 = np.array([u1, v1, 1])
        m2 = np.array([u2, v2, 1])

        l1 = F.T @ m2
        l2 = F @ m1

        ax1.scatter(u1, v1)
        ax2.scatter(u2, v2)

        show_line(l1, width, height, ax1)
        show_line(l2, width, height, ax2)


def show_camera_3d(ax, P, text=''):
    R, C = P[:, :3], -P[:, 3]
    C[-1] *= -1

    ax.quiver(*C, *(R.T @ ([1, 0, 0])), color='b', arrow_length_ratio=0.2)
    ax.quiver(*C, *(R.T @ ([0, 1, 0])), color='g', arrow_length_ratio=0.2)
    ax.quiver(*C, *(R.T @ ([0, 0, 1])), color='r', arrow_length_ratio=0.2)
    
    ax.text(*(C + -1e-1), text)
    ax.scatter(*C, c='k')


def create_3d_plot(plt):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-1, 5)

    return fig, ax


def show_point_cloud(X, ax):
    ax.scatter(X[0], X[1], X[2], marker='.', s=3)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-1, 5)

    R = Ry(np.pi / 4)
    P1 = np.c_[np.diag([1, 1 ,1]), [0, 0, 0]]
    P2 = np.c_[R, [1, 0, 1]]
    P2 = np.load('P2.npy')

    show_camera_3d(ax, P1)
    show_camera_3d(ax, P2)

    plt.show()
