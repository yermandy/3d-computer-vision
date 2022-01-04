import rectify
import scipy.io
import pickle
from lib import *
from plots import *
from PIL import Image
import matplotlib.pyplot as plt
import corresp
import os

pairs = [
    # horizontal
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    # vertical
    # (1, 5), (2, 6), (3, 7), (4, 8),
    # (5, 9), (6, 10), (7, 11), (8, 12)
]

scene_root = 'scene'

K = np.loadtxt(f'{scene_root}/K.txt')

with open('cache/cameras.pickle', 'rb') as f:
    cameras = pickle.load(f)

with open('cache/c.pickle', 'rb') as f:
    c: corresp.Corresp = pickle.load(f)

task = []

if not os.path.isfile('stereo_out.mat'):
    for c1, c2 in pairs:
        P1 = K @ cameras[c1].P
        P2 = K @ cameras[c2].P
        F = P2F(P1, P2)

        X1, u1, _ = c.get_Xu(c1)
        X2, u2, _ = c.get_Xu(c2)

        X1, idx = np.unique(X1, return_index=True)
        u1 = u1[idx]

        X2, idx = np.unique(X2, return_index=True)
        u2 = u2[idx]

        intersection = np.intersect1d(X1, X2)

        mask1 = np.isin(X1, intersection, assume_unique=True)
        mask2 = np.isin(X2, intersection, assume_unique=True)
        
        indices1 = u1[mask1]
        indices2 = u2[mask2]

        u1 = cameras[c1].features[indices1]
        u2 = cameras[c2].features[indices2]

        img_1 = np.array(Image.open(f'scene/images/{c1:02d}.jpg'))
        img_2 = np.array(Image.open(f'scene/images/{c2:02d}.jpg'))

        # region debug points

        # fig, axs = plt.subplots(1, 2, figsize=(10,5))
        # axs[0].imshow(img_1, cmap='gray')
        # axs[1].imshow(img_2, cmap='gray')
        # for i in range(0, len(u1), 50):
        #     x, y = u1[i]
        #     axs[0].scatter(x, y, marker='.')
        #     x, y = u2[i]
        #     axs[1].scatter(x, y, marker='.')
        # plt.tight_layout()
        # plt.show()

        # endregion

        H1, H2, img_1_rect, img_2_rect = rectify.rectify(F, img_1, img_2)
        
        u1_r = H1 @ e2p(u1.T)
        u1_r = p2e(u1_r)

        u2_r = H2 @ e2p(u2.T)
        u2_r = p2e(u2_r)

        # region debug homographies

        # height, width = img_1_rect.shape
        # u1_r = u1_r.T
        # u2_r = u2_r.T
        # fig, axs = plt.subplots(1, 2, figsize=(10,5))
        # axs[0].imshow(img_1_rect, cmap='gray')
        # axs[1].imshow(img_2_rect, cmap='gray')
        # indices = np.random.choice(range(len(u1_r)), size=10, replace=False)
        # for i in indices:
        #     x, y = u1_r[i]
        #     axs[0].scatter(x, y, marker='.')
        #     axs[0].plot([0, width], [y, y])
        #     x, y = u2_r[i]
        #     axs[1].scatter(x, y, marker='.')
        #     axs[1].plot([0, width], [y, y])
        # plt.tight_layout()
        # plt.show()

        # endregion
        
        seeds = np.vstack((u1_r[0], u2_r[0], (u1_r[1] + u2_r[1]) / 2)).T
        task_i = np.array([img_1_rect, img_2_rect, seeds], dtype=object)
        task += [ task_i ]

    task = np.vstack(task)
    scipy.io.savemat('stereo_in.mat', {'task': task})


D = scipy.io.loadmat('stereo_out.mat')
D = D['D']

X_all = []
colors_all = []

# for i, (c1, c2) in enumerate([(1, 2), (2, 3), (3, 4)]):
for i, (c1, c2) in enumerate(pairs):
    print('pair:', c1, c2)

    P1 = K @ cameras[c1].P
    P2 = K @ cameras[c2].P
    F = P2F(P1, P2)

    img_1 = np.array(Image.open(f'scene/images/{c1:02d}.jpg'))
    img_2 = np.array(Image.open(f'scene/images/{c2:02d}.jpg'))
    H1, H2, img_1_rect, img_2_rect = rectify.rectify(F, img_1, img_2)

    Di = D[i, 0]
    not_nan = ~np.isnan(Di)
    y, x = np.where(not_nan)
    u1 = np.array([x, y])
    u2 = np.array([x + Di[y, x], y])

    # region disparities
    # fig, axs = plt.subplots(1, 3, figsize=(20,10))
    # Di[np.isnan(Di)] = 0
    # axs[0].imshow(img_1_rect, cmap='gray')
    # axs[1].imshow(Di)
    # axs[2].imshow(img_2_rect, cmap='gray')
    # plt.tight_layout()
    # plt.show()
    # endregion 

    # region debug new rectified correspondences
    # u1 = u1.T
    # u2 = u2.T
    # fig, axs = plt.subplots(1, 2, figsize=(20,10))
    # axs[0].imshow(img_1_rect, cmap='gray')
    # axs[1].imshow(img_2_rect, cmap='gray')
    # indices = np.random.choice(range(len(u1)), size=200, replace=False)
    # for i in indices:
    #     x, y = u1[i]
    #     axs[0].scatter(x, y, marker='.')
    #     x, y = u2[i]
    #     axs[1].scatter(x, y, marker='.')
    # plt.tight_layout()
    # plt.show()
    # endregion debug 

    u1 = e2p(u1)
    u2 = e2p(u2)
 
    u1 = p2e(np.linalg.inv(H1) @ u1)
    u2 = p2e(np.linalg.inv(H2) @ u2)

    # region debug new correspondences after undo recification
    # u1 = u1.T
    # u2 = u2.T
    # fig, axs = plt.subplots(1, 2, figsize=(20,10))
    # axs[0].imshow(img_1, cmap='gray')
    # axs[1].imshow(img_2, cmap='gray')
    # indices = np.random.choice(range(len(u1)), size=200, replace=False)
    # for i in indices:
    #     x, y = u1[i]
    #     axs[0].scatter(x, y, marker='.')
    #     x, y = u2[i]
    #     axs[1].scatter(x, y, marker='.')
    # plt.tight_layout()
    # plt.show()
    # endregion

    correspondences = []
    colors = []
    for (x1, y1), (x2, y2) in zip(u1.T, u2.T):
        correspondences.append([x1, y1, 1, x2, y2, 1])
        # color = (img_1[round(y1), round(x1)] + img_2[round(y2), round(x2)]) / 2
        color = img_1[int(y1), int(x1)]
        colors.append(color)
    colors = np.array(colors).T
    correspondences = np.array(correspondences)

    inliers, X = reconstruct_point_cloud_2(correspondences, P1, P2, theta=1)

    colors = colors[:, inliers]
    # colors_all.append(colors[:, inliers])

    mask_1 = X <= 15
    mask_2 = X >= -15
    mask = np.sum(mask_1 & mask_2, axis=0) == 3
    X = X[:, mask]
    colors = colors[:, mask]
    colors_all.append(colors)

    
    # region show dense point cloud
    # fig, ax = create_3d_plot(plt)
    # show_point_cloud(X, ax)
    # plt.show()
    # endregion

    X_all.append(X)

X_all = np.hstack(X_all)
colors_all = np.clip(np.round(np.hstack(colors_all)), 0, 255).astype(np.uint8)

print(X_all.shape)
print(colors_all.shape)

import ge
g = ge.ge.GePly('X.ply')
g.points(X_all, colors_all)
g.close()