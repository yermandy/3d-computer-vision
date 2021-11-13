import matplotlib.pyplot as plt
import numpy as np
import math

np.set_printoptions(suppress=True)


def Rx(alpha):
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    return np.array([
        [1, 0, 0],
        [0, cos_alpha, -sin_alpha],
        [0, sin_alpha, cos_alpha]
    ])


def Ry(alpha):
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    return np.array([
        [cos_alpha, 0, sin_alpha],
        [0, 1, 0],
        [-sin_alpha, 0, cos_alpha]
    ])


def Rz(alpha):
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    return np.array([
        [cos_alpha, -sin_alpha, 0],
        [sin_alpha, cos_alpha, 0],
        [0, 0, 1]
    ])


def projective_to_euclidean(P):
    E = np.empty((2, P.shape[1]))
    E[0] = P[0] / P[2]
    E[1] = P[1] / P[2]
    return E


def plot(UV1, UV2):
    UV1 = projective_to_euclidean(UV1)
    UV2 = projective_to_euclidean(UV2)

    u1, v1 = UV1
    u2, v2 = UV2

    plt.figure()
    plt.plot(u1, v1, 'r-', linewidth=2)
    plt.plot(u2, v2, 'b-', linewidth=2)
    plt.plot([u1, u2], [v1, v2], 'k-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.axis('equal')


X1 = [ [-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,  0,  0.5 ],
       [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5 ],
       [ 4,    4,   4,    4,    4,    4,    4,    4,    4,    4,  4   ] ]

X2 = [ [-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,  0,    0.5 ],
       [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1,   -0.5 ],
       [ 4.5,  4.5, 4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5 ] ]

X1 = np.array(X1)
X2 = np.array(X2)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-1, 5)

ax.plot3D(X1[0], X1[1], X1[2], c='r')
ax.plot3D(X2[0], X2[1], X2[2], c='b')

for i in range(X2.shape[1]):
    X = np.vstack([X1[:, i], X2[:, i]])
    ax.plot3D(X[:, 0], X[:, 1], X[:, 2], c='k')


K = [[ 1000,    0, 500 ],
     [    0, 1000, 500 ],
     [    0,    0,   1 ]]
K = np.array(K)

# x, y, z = K[:, 2]

unit_length = 1
## z-axis
ax.plot3D([0, 0], [0, 0], [0, unit_length], c='r')
## y-axis
ax.plot3D([0, 0], [0, unit_length], [0, 0], c='g')
## x-axis
ax.plot3D([0, unit_length], [0, 0], [0, 0], c='b')


I = np.identity(3)
## R1 = R2 = R3 = I

X1 = np.vstack([X1, np.ones(X1.shape[1])])
X2 = np.vstack([X2, np.ones(X2.shape[1])])

## Task 1. Camera in the origin looking in the direction of z-axis.
C1 = np.array([0, 0, 0])
P1 = K @ np.c_[I, C1]

UV1 = P1 @ X1
UV2 = P1 @ X2
# ax.scatter(*C1, c='k', marker='x')
plot(UV1, UV2)

## Task 2. Camera located at [0, -1 ,0] looking in the direction of z-axis.
C2 = np.array([0, -1, 0])
P2 = K @ np.c_[I, -C2]

UV1 = P2 @ X1
UV2 = P2 @ X2
# ax.scatter(*C2, c='k', marker='x')
plot(UV1, UV2)

## Task 3. Camera located at [0, 0.5 ,0] looking in the direction of z-axis.
C3 = np.array([0, 0.5, 0])
P3 = K @ np.c_[I, -C3]

UV1 = P3 @ X1
UV2 = P3 @ X2
# ax.scatter(*C3, c='k', marker='x')
plot(UV1, UV2)

## Task 4. Camera located at [0, -3, 0.5], with optical axis rotated by 0.5 rad around x-axis towards y-axis.
C4 = np.array([0, -3, 0.5])
R4 = Rx(0.5)
P4 = K @ R4 @ np.c_[I, -C4]

UV1 = P4 @ X1
UV2 = P4 @ X2
# ax.scatter(*C4, c='k', marker='x')
plot(UV1, UV2)

## Task 5. Camera located at [0, -5, 4.2] looking in the direction of y-axis.
C5 = np.array([0, -5, 4.2])
R5 = Rx(np.pi / 2)
P5 = K @ R5 @ np.c_[I, -C5]

UV1 = P5 @ X1
UV2 = P5 @ X2
# ax.scatter(*C5, c='k', marker='x')
plot(UV1, UV2)

## Task 6. Camera located at [-1.5, -3, 1.5], with optical axis rotated by 0.5 rad around y-axis 
## towards x-axis (i.e., -0.5 rad) followed by a rotation by 0.8 rad around x-axis towards y-axis.
C6 = np.array([-1.5, -3, 1.5])
R6 = Rx(0.8) @ Ry(-0.5)
P6 = K @ R6 @ np.c_[I, -C6]

UV1 = P6 @ X1
UV2 = P6 @ X2
# ax.scatter(*C6, c='k', marker='x')
plot(UV1, UV2)

plt.show()