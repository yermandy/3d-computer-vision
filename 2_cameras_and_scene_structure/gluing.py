from plots import *
from lib import *    
from utils import *    
import corresp
import os
import pickle
from easydict import EasyDict as dict

os.makedirs('cache', exist_ok=True)

scene_root = 'scene'

def initialize_c(N=12, verbose=0):
    c = corresp.Corresp(N + 1)
    c.verbose = verbose

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            matches = np.loadtxt(f'{scene_root}/matches/m_{i:02d}_{j:02d}.txt').astype(int)
            c.add_pair(i, j, matches)

    return c

image_id_1 = 7
image_id_2 = 11
c = initialize_c()
K = np.loadtxt(f'{scene_root}/K.txt')

matches = np.array(c.get_m(image_id_1, image_id_2)).T

points_1 = np.loadtxt(f'{scene_root}/matches/u_{image_id_1:02d}.txt')
points_2 = np.loadtxt(f'{scene_root}/matches/u_{image_id_2:02d}.txt')

correspondences = get_correspondences(matches, points_1, points_2)
calibrated_correspondences = calibrate_correspondences(correspondences, K)

if not os.path.isfile('cache/E.npy') or not os.path.isfile('cache/P2.npy') or not os.path.isfile('cache/inliers.npy'):
    P2, E, inliers = p5_ransac(calibrated_correspondences, 100)
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


cameras = {}

for i in range(1, 13):
    cameras[i] = dict()
    cameras[i].features = np.loadtxt(f'{scene_root}/matches/u_{i:02d}.txt')

cameras[7].P = P1
cameras[11].P = P2
cameras[7].X_idx = np.arange(0, X.shape[1])

for i in range(10):
    cameras_next, cameras_points = c.get_green_cameras()

    camera_i = cameras_next[np.argmax(cameras_points)]

    X2U_idx = np.array(c.get_Xu(camera_i)[0:2])

    print(f'camera {camera_i} p3p')

    u3 = np.loadtxt(f'{scene_root}/matches/u_{camera_i:02d}.txt').T
    R3, t3, inliers3 = p3p_ransac(X, u3, X2U_idx, K)
    P_i = np.c_[R3, t3]
    cameras[camera_i].P = P_i

    new_inliers_indices = np.flatnonzero(inliers3)

    c.join_camera(camera_i, new_inliers_indices)

    n_X_from = X.shape[1]

    points_2 = cameras[camera_i].features

    for camera_j in c.get_cneighbours(camera_i):

        print(f'\ntriangulation {camera_i} -> {camera_j}')
        
        P_j = cameras[camera_j].P

        c_idx_i, c_idx_j = np.array(c.get_m(camera_i, camera_j))

        points_1 = cameras[camera_j].features

        matches = np.c_[c_idx_j, c_idx_i]
        correspondences = get_correspondences(matches, points_1, points_2, projective=True)

        inliers, Xj = reconstruct_point_cloud_2(correspondences, K @ P_j, K @ P_i, theta=2)

        print('inliers: ', inliers.sum())

        X = np.hstack((X, Xj))

        inliers = np.flatnonzero(inliers)

        c.new_x(camera_i, camera_j, inliers)

    n_X_till = X.shape[1]
        
    ilist = c.get_selected_cameras();

    for camera_j in c.get_selected_cameras():

        X_idx, u_idx, Xu_verified = c.get_Xu(camera_j)
        Xu_tentative = ~Xu_verified
        
        if Xu_tentative.sum() == 0:
            continue

        print(f'\nverification {camera_j}')

        Xu_tentative_idx = np.flatnonzero(Xu_tentative)

        X_idx, u_idx = X_idx[Xu_tentative], u_idx[Xu_tentative]

        u_true = cameras[camera_j].features[u_idx]
        up_true = e2p(u_true.T)

        X_j = X[:, X_idx]
        X_j = e2p(X_j)

        P_j = cameras[camera_j].P
        P_j = K @ P_j
                
        up_pred = P_j @ X_j

        errors = euclidean_reprojection_error(up_true, up_pred)

        threshold = 1
        
        mask = errors <= threshold
        
        inliers = Xu_tentative_idx[mask]

        print(f'inliers {camera_j}: {np.sum(mask)}')
        
        c.verify_x(camera_j, inliers)
    
    cameras[camera_i].X_idx = np.arange(n_X_from, n_X_till)
    c.finalize_camera()
# exit()

with open('cache/cameras.pickle', 'wb') as f:
    pickle.dump(cameras, f)

with open('cache/c.pickle', 'wb') as f:
    pickle.dump(c, f)

fig, ax = create_3d_plot(plt)

X_save = []

for idx, camera in cameras.items():
    show_camera_3d(ax, camera.P, idx)
    if 'X_idx' in camera:
        X_j = X[:, camera.X_idx]
        mask_1 = X_j <= 15
        mask_2 = X_j >= -15
        mask = np.sum(mask_1 & mask_2, axis=0) == 3
        X_j = X_j[:, mask]
        X_save.append(X_j)
        show_point_cloud(X_j, ax)
X_save = np.hstack(X_save)

ax.view_init(-90, -90)
plt.tight_layout()
plt.subplots_adjust(0, 0, 1, 1)
plt.show()


save_points(X_save, cameras)