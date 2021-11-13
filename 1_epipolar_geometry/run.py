from plots import *
from lib import *    

image_id_1 = 1
image_id_2 = 2

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
P2, E, inliers = epipolar_ransac(calibrated_correspondences, 100)

K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes

show_image(image_1, ax1)
show_image(image_2, ax2)

show_epipolar_lines(correspondences[inliers], F, ax1, ax2, width, height, each=100)

# show_needle_map(correspondences[inliers], c='tab:red', ax=ax1)
# show_needle_map(inverse_correspondences[inliers], c='tab:blue', ax=ax2)

plt.tight_layout()
plt.show()