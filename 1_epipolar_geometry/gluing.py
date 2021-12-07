from numpy import percentile
from plots import *
from lib import *    
import corresp
import os
from easydict import EasyDict as dict

def initialize_c(N=12, verbose=1):
    c = corresp.Corresp(N + 1)
    c.verbose = verbose

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            matches = np.loadtxt(f'scene/matches/m_{i:02d}_{j:02d}.txt').astype(int)
            c.add_pair(i, j, matches)

    return c

image_id_1 = 7
image_id_2 = 11
c = initialize_c()
K = np.loadtxt(f'scene/K.txt')

matches = np.array(c.get_m(image_id_1, image_id_2)).T

image_1 = plt.imread(f'scene/images/{image_id_1:02d}.jpg')
image_2 = plt.imread(f'scene/images/{image_id_2:02d}.jpg')
height, width = image_1.shape[0], image_1.shape[1]

points_1 = np.loadtxt(f'scene/matches/u_{image_id_1:02d}.txt')
points_2 = np.loadtxt(f'scene/matches/u_{image_id_2:02d}.txt')


correspondences = get_correspondences(matches, points_1, points_2)
calibrated_correspondences = calibrate_correspondences(correspondences, K)

if not os.path.isfile('cache/E.npy') or not os.path.isfile('cache/P2.npy') or not os.path.isfile('cache/inliers.npy'):
    P2, E, inliers = p5_ransac(calibrated_correspondences, 25, theta=1e-4, K=K)
    # np.save('cache/P2.npy', P2)
    # np.save('cache/E.npy', E)
    # np.save('cache/inliers.npy', inliers)
else:
    P2 = np.load('cache/P2.npy')
    E = np.load('cache/E.npy')
    inliers = np.load('cache/inliers.npy')

P1 = get_P1()
F = calc_F(K, E)

X, inliers = reconstruct_point_cloud(calibrated_correspondences, inliers, P1, P2)

inliers_idx = np.flatnonzero(inliers)
c.start(image_id_1, image_id_2, inliers_idx)

# print('c.get_Xu(3)')
# print(c.get_Xu(3))
# X2U_idx


next_camera_id = 3

X2U_idx = np.array(c.get_Xu(next_camera_id)[0:2])
print(X2U_idx)
print(X2U_idx.shape)
# print(u3.shape)
u3 = np.loadtxt(f'scene/matches/u_{next_camera_id:02d}.txt').T
R3, t3, inliers3 = p3p_ransac(X, u3, X2U_idx, K)
P3 = np.c_[R3, t3]

cameras = {}
cameras[7] = dict({'P': P1})
cameras[11] = dict({'P': P2})

cameras[next_camera_id] = dict({'P': P3})


new_inliers_indices = np.flatnonzero(inliers3)

c.join_camera(next_camera_id, new_inliers_indices)

verified_cameras = c.get_cneighbours(next_camera_id)

X_from_two = X.copy()

P_j = P3

for verified_camera_id in verified_cameras:
    

    P_i = cameras[verified_camera_id].P

    c_idx_i, c_idx_j = np.array(c.get_m(next_camera_id, verified_camera_id))

    print('correspondences')
    print(correspondences.shape)
    # calibrated_correspondences = calibrate_correspondences(correspondences, K)

    # print(calibrated_correspondences.shape)
    print('inliers3.shape')
    print(inliers3.shape)


    points_1 = np.loadtxt(f'scene/matches/u_{verified_camera_id:02d}.txt')
    points_2 = np.loadtxt(f'scene/matches/u_{next_camera_id:02d}.txt')

    matches = np.c_[c_idx_j, c_idx_i]
    correspondences = get_correspondences(matches, points_1, points_2, projective=True)

    inliers, X_new = reconstruct_point_cloud_2(correspondences, K @ P_i, K @ P_j, theta=2)
    # Xj = reconstruct_point_cloud(calibrated_correspondences, inliers3, P1, P2, corretion=True)

    X = np.hstack((X, X_new))

    inliers = np.flatnonzero(inliers)

    c.new_x(next_camera_id, verified_camera_id, inliers)

# exit()


cameras_next, cameras_points = c.get_green_cameras()
# exit()
# print(c.get_cneighbours(next_camera_id))

# c.verify_x(next_camera_id, new_inliers_indices)

# c.finalize_camera()

# print(X2U_idx[1])
# print(new_inliers)

print()

# exit()

print(X.shape)

fig, ax = create_3d_plot(plt)
show_point_cloud(X[:, len(X_from_two):], ax)
show_point_cloud(X_from_two, ax)

show_camera_3d(ax, P1, 7)
show_camera_3d(ax, P2, 11)
show_camera_3d(ax, P3, 3)


ax.view_init(-90, -90)
plt.tight_layout()
plt.subplots_adjust(0, 0, 1, 1)
plt.show()