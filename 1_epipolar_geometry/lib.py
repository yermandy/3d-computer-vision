import numpy as np
import p5
import math


def line_from_points(x1, y1, x2, y2):
    return np.cross([x1, y1, 1], [x2, y2, 1])


def lines_intersection(l1, l2):
    x = np.cross(l1, l2)
    x = [x[0] / x[2], x[1] / x[2]]
    return x


def get_correspondences(matches, points_1, points_2):
    correspondences = []
    for i, j in matches:
        x1, y1 = points_1[i]
        x2, y2 = points_2[j]
        correspondences.append([x1, y1, x2, y2])
    correspondences = np.array(correspondences)
    return correspondences


def calibrate_correspondences(correspondences, K):  
    u1, u2 = correspondences[:, [0, 1]], correspondences[:, [2, 3]]
    u1p = np.c_[u1, np.ones(len(u1))]
    u2p = np.c_[u2, np.ones(len(u2))]
    K_inv = np.linalg.inv(K)

    u1p = (K_inv @ u1p.T).T
    u2p = (K_inv @ u2p.T).T
    
    calibrated_correspondences = np.c_[u1p, u2p]
    return calibrated_correspondences


def zero_one_support(distances, theta):
    inliers = distances <= theta
    support = inliers.sum()
    return support, inliers


def mle_support(distances, theta):
    inliers = distances <= theta
    inliers_distances = distances[inliers]
    support = np.sum(1 - np.power(inliers_distances, 2) / np.power(theta, 2))
    return support, inliers


def sqc(x):
    assert x.shape == (3, )
    x1, x2, x3 = x
    return np.array([
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ])


def p2e(X):
    #  in shape = (D, N)
    # out shape = (D - 1, N)
    X = np.array(X)
    return X[:-1] / X[-1]


def Pu2X(P1, P2, u1p, u2p):
    assert P1.shape == P2.shape == (3, 4)
    assert u1p.shape == u2p.shape

    _, n_samples = u1p.shape

    p11, p12, p13 = P1
    p21, p22, p23 = P2

    X = []

    for i in range(n_samples):
        u1i, u2i = u1p[:, i], u2p[:, i]
        u1, v1, _ = u1i
        u2, v2, _ = u2i

        D = np.array([
            u1 * p13 - p11,
            v1 * p13 - p12,
            u2 * p23 - p21,
            v2 * p23 - p22
        ])

        # TODO numerical conditioning
        # print(np.max(D) - np.min(D))

        _, _, V_t = np.linalg.svd(D)

        x = V_t[-1]
        X.append(x)

    X = np.transpose(X)
    return X


def Eu2Rt(E, u1p, u2p):
    # Essential matrix decomposition with cheirality constraint: 
    #   all 3D points are in front of both cameras
    #   see 9.6.3 in book
    assert E.shape == (3, 3)
    assert u1p.shape[0] == 3
    assert u2p.shape[0] == 3

    U, D, V_t = np.linalg.svd(E)
    
    R = np.diag([1, 1 ,1])
    t = np.array([0, 0, 0])

    P1 = np.c_[R, t]
    p11, p12, p13 = P1

    for a in [-1, 1]:
        for b in [-1, 1]:
            W = np.array([
                [0, a, 0],
                [-a, 0, 0],
                [0, 0, 1]
            ])
            
            R = U @ W @ V_t
            t = b * U[:, -1]
            P2 = np.c_[R, t]

            # cheirality constraint https://cmsc426.github.io/sfm/#tri
            X = Pu2X(P1, P2, u1p, u2p)
            X = p2e(X)
            cheirality = R[2] @ (X - t.reshape(-1, 1))

            if np.all(cheirality > 0):
                return R, t
            
    return R, t


def reprojection_error(UV, P1, P2):
    u1p = UV[:, :3].T
    u2p = UV[:, 3:].T
    
    u1, v1, u2, v2 = u1p[0], u1p[1], u2p[0], u2p[1]
    
    p11, p12, p13 = P1
    p21, p22, p23 = P2

    X = Pu2X(P1, P2, u1p, u2p)

    p13X = p13 @ X
    p23X = p23 @ X

    e1 = (u1 - (p11 @ X) / p13X) ** 2 + (v1 - (p12 @ X) / p13X) ** 2
    e2 = (u2 - (p21 @ X) / p23X) ** 2 + (v2 - (p22 @ X) / p23X) ** 2
    e = e1 + e2

    return e

def calc_F(K, E):
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


def epipolar_ransac(correspondences, n_iters, support_function=mle_support, theta=0.00001, n_samples=5):
    N = len(correspondences)
    support_best = 0
    P2_best = None
    E_best = None
    inliers_best = None
    
    for _ in range(n_iters):
        indices = np.random.choice(N, n_samples, replace=False)
        sample = correspondences[indices]

        u1p = sample[:, 0:3].T
        u2p = sample[:, 3:6].T

        Es = p5.p5gb(u1p, u2p)

        for E in Es:            
            R, t = Eu2Rt(E, u1p, u2p)

            # calculate reprojection error
            P1 = np.c_[np.diag([1, 1 ,1]), [0, 0, 0]]
            P2 = np.c_[R, t]
            
            error = reprojection_error(correspondences, P1, P2)
            support, inliers = support_function(error, theta)
            
            if support > support_best:
                support_best = support
                P2_best = P2
                E_best = E
                inliers_best = inliers
                print(support_best)

    return P2_best, E_best, inliers_best

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
