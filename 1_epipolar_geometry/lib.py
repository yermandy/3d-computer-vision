import numpy as np
import p5
import p3p
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


def calibrate_correspondences(correspondences, K, normalize=False):  
    u1, u2 = correspondences[:, [0, 1]], correspondences[:, [2, 3]]
    u1p = np.c_[u1, np.ones(len(u1))]
    u2p = np.c_[u2, np.ones(len(u2))]
    K_inv = np.linalg.inv(K)

    u1p = (K_inv @ u1p.T).T
    u2p = (K_inv @ u2p.T).T

    if normalize:
        u1p /= u1p[:, 2].reshape(-1, 1)
        u2p /= u2p[:, 2].reshape(-1, 1)
    
    calibrated_correspondences = np.c_[u1p, u2p]
    return calibrated_correspondences


def calibrate_features(u, K):
    u_p = e2p(u)
    return np.linalg.inv(K) @ u_p


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


def e2p(X):
    return np.vstack((X, np.ones((1, X.shape[1]))))


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

    for a in [-1, 1]:
        for b in [-1, 1]:
            W = np.array([
                [0, a, 0],
                [-a, 0, 0],
                [0, 0, 1]
            ])
            
            R = U @ W @ V_t
            t = b * U[:, -1]
            
            # Should it be here?
            if np.linalg.det(R) < 0:
                R = -R
                t = -t

            P2 = np.c_[R, t]

            # cheirality constraint https://cmsc426.github.io/sfm/#tri
            X = Pu2X(P1, P2, u1p, u2p)
            X /= X[-1]

            if np.any(X[2] < 0):
                continue
            
            X = p2e(X)
            cheirality = R[2] @ (X - t.reshape(-1, 1))

            if np.all(cheirality > 0):
                return R, t
    
    return None, None


def reprojection_error(UV, P1, P2, return_X=False):
    u1p = UV[:, :3].T
    u2p = UV[:, 3:].T
    
    u1, v1, u2, v2 = u1p[0], u1p[1], u2p[0], u2p[1]
    
    p11, p12, p13 = P1
    p21, p22, p23 = P2

    X = Pu2X(P1, P2, u1p, u2p)

    # what if we normalize it?
    # X /= X[-1]

    p13X = p13 @ X
    p23X = p23 @ X

    e1 = (u1 - (p11 @ X) / p13X) ** 2 + (v1 - (p12 @ X) / p13X) ** 2
    e2 = (u2 - (p21 @ X) / p23X) ** 2 + (v2 - (p22 @ X) / p23X) ** 2
    e = e1 + e2

    if return_X:
        return e, X

    return e


def P2F(P1, P2):
    Q1, q1 = P1[:, :-1], P1[:, -1]
    Q2, q2 = P2[:, :-1], P2[:, -1]
    Q1_Q2_inv = Q1 @ np.linalg.inv(Q2)
    F = Q1_Q2_inv.T @ sqc(q1 - Q1_Q2_inv @ q2)
    return F


def sampson_error(UV, P1, P2, return_vectors=False):
    F = P2F(P1, P2)

    x = UV[:, :3]
    y = UV[:, 3:]

    S = [[1, 0, 0], [0, 1, 0]]

    SF_t = S @ F.T
    SF = S @ F

    sampson_errors = []
    sampson_vectors = []
    for x_i, y_i in zip(x, y):
        # epipolar algebraic error
        e_i = y_i @ F @ x_i
        J_i = np.concatenate((SF_t @ y_i, SF @ x_i))

        sampson_vector = -(J_i * e_i) / np.sum(J_i ** 2)
        sampson_error = np.linalg.norm(sampson_vector)
        
        sampson_errors.append(sampson_error)
        sampson_vectors.append(sampson_vector)
    
    sampson_errors = np.array(sampson_errors)
    sampson_vectors = np.array(sampson_vectors)
    
    if return_vectors:
        return sampson_errors, sampson_vectors

    return sampson_errors


def calc_F(K, E):
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


def inliers_in_front_camera(correspondences, inliers, P1, P2):
    u1p = correspondences[:, 0:3].T
    u2p = correspondences[:, 3:6].T

    X = Pu2X(P1, P2, u1p, u2p)
    X = p2e(X)

    mask_in_front = X[2] > 0
    inliers = inliers & mask_in_front
    
    return inliers


def p5_ransac(correspondences, n_iters, support_function=mle_support, theta=1e-4, n_samples=5, K=None):
    N = len(correspondences)
    support_best = 0
    P2_best = None
    E_best = None
    inliers_best = None

    P1 = get_P1()
    
    for _ in range(n_iters):
        indices = np.random.choice(N, n_samples, replace=False)
        sample = correspondences[indices]

        u1p = sample[:, 0:3].T
        u2p = sample[:, 3:6].T

        Es = p5.p5gb(u1p, u2p)

        for E in Es:            
            R, t = Eu2Rt(E, u1p, u2p)

            if R is None:
                continue

            P2 = np.c_[R, t]
            
            # calculate sampson error
            error = sampson_error(correspondences, P1, P2)

            # calculate reprojection error
            # error = reprojection_error(correspondences, P1, P2)

            support, inliers = support_function(error, theta)
            
            if support > support_best:
                support_best = support
                P2_best = P2
                E_best = E
                inliers_best = inliers
                print(support_best)

    inliers_best = inliers_in_front_camera(correspondences, inliers_best, P1, P2)

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


def get_P1():
    P1 = np.c_[np.diag([1, 1 ,1]), [0, 0, 0]]
    return P1


def reconstruct_point_cloud(correspondences, inliers, P1, P2, corretion=True):
    u1p = correspondences[:, 0:3].T
    u2p = correspondences[:, 3:6].T

    if corretion:
        # The Golden Standard Method
        _, sampson_vectors = sampson_error(correspondences, P1, P2, True)
        for i, correction_vector in enumerate(sampson_vectors):
            u1p[[0, 1], i] -= correction_vector[:2]
            u2p[[0, 1], i] -= correction_vector[2:]

    X = Pu2X(P1, P2, u1p, u2p)
    X = p2e(X)

    mask_in_front = X[2] > 0
    new_inliers = inliers & mask_in_front
    
    X = X[:, new_inliers]

    return X, new_inliers


def reconstruct_point_cloud_2(correspondences, P1, P2, theta=1e-5, corretion=True):
    u1p = correspondences[:, 0:3].T
    u2p = correspondences[:, 3:6].T

    if corretion:
        # The Golden Standard Method
        _, sampson_vectors = sampson_error(correspondences, P1, P2, True)
        for i, correction_vector in enumerate(sampson_vectors):
            u1p[[0, 1], i] -= correction_vector[:2]
            u2p[[0, 1], i] -= correction_vector[2:]

    correspondences = np.r_[u1p, u2p].T

    errors, X = reprojection_error(correspondences, P1, P2, return_X=True)

    X = p2e(X)

    inliers = errors < theta

    # print(errors)

    print('inliers sum ', inliers.sum())

    return inliers, X[:, inliers]


def p3p_ransac(X, u, X2U_idx, K, p=0.99999, theta=2):
    up_K = calibrate_features(u, K)
    up = e2p(u)
    
    X = e2p(X)
    
    N = X2U_idx.shape[1]
    
    support_best = 0
    R_best = None
    t_best = None
    inliers_best = None

    N_max = 1000
    N_iter = 0

    X_inliers = X[:, X2U_idx[0]]
    up_inliers = up[:, X2U_idx[1]]

    while N_iter <= N_max:
        indices = np.random.choice(N, 3, replace=False)
        X_idx = X2U_idx[0, indices]
        U_idx = X2U_idx[1, indices]

        X_i = X[:, X_idx]
        upK_i = up_K[:, U_idx]

        solutions = p3p.p3p_grunert(X_i, upK_i)

        for X_j in solutions:
            R_j, t_j = p3p.XX2Rt_simple(X_i, X_j)
            P_j = K @ np.c_[R_j, t_j]
            
            up_pred = P_j @ X_inliers

            # select only points that are in front of the camera
            # inliers_front = up_pred[2] > 0
            # errors = euclidean_reprojection_error(up_inliers[:, inliers_front], up_pred[:, inliers_front])

            errors = euclidean_reprojection_error(up_inliers, up_pred)

            support, inliers = mle_support(errors, theta)

            if support > support_best:
                support_best = support
                R_best = R_j
                t_best = t_j
                inliers_best = inliers

                print(support_best)

                # eps = 1 - np.mean(inliers)
                # N_max = 0 if eps < 1e-8 else math.log(1 - p) / math.log(1 - (1 - eps) ** 3)


        N_iter += 1

    return R_best, t_best, inliers_best


def euclidean_reprojection_error(u_true, u_pred):
    u_true = p2e(u_true)
    u_pred = p2e(u_pred)
    diffs = u_pred - u_true
    errors = np.sum(diffs ** 2, axis=0)
    return errors