from plots import *
from lib import *    

image_id_1 = 1
image_id_2 = 4

image_1 = plt.imread(f'scene/images/{image_id_1:02d}.jpg')
image_2 = plt.imread(f'scene/images/{image_id_2:02d}.jpg')
height, width = image_1.shape[0], image_1.shape[1]

points_1 = np.loadtxt(f'scene/matches/u_{image_id_1:02d}.txt')
points_2 = np.loadtxt(f'scene/matches/u_{image_id_2:02d}.txt')
matches = np.loadtxt(f'scene/matches/m_{image_id_1:02d}_{image_id_2:02d}.txt').astype(int)
K = np.loadtxt(f'scene/K.txt')


correspondences = get_correspondences(matches, points_1, points_2)
inverse_correspondences = correspondences[:, [2, 3, 0, 1]]
calibrated_correspondences = calibrate_correspondences(correspondences, K)
P2, E, inliers = epipolar_ransac(calibrated_correspondences, 50)
F = calc_F(K, E)

u1p = calibrated_correspondences[inliers, 0:3].T
u2p = calibrated_correspondences[inliers, 3:6].T

P1 = np.c_[np.diag([1, 1 ,1]), [0, 0, 0]]
X = Pu2X(P1, P2, u1p, u2p)
X = p2e(X)

fig, axes = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axes.flatten()

show_image(image_1, ax1)
show_image(image_2, ax2)
show_image(image_1, ax3)
show_image(image_2, ax4)

show_epipolar_lines(correspondences[inliers], F, ax1, ax2, width, height, n_lines=15)

show_needle_map(correspondences[inliers], c='tab:red', ax=ax3)
show_needle_map(inverse_correspondences[inliers], c='tab:blue', ax=ax4)

plt.tight_layout()
# plt.show()

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-1, 5)

ax.scatter(X[0], X[1], X[2], marker='.')

show_camera_3d(ax, P1)
show_camera_3d(ax, P2)

plt.show()