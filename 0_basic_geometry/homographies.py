import matplotlib.pyplot as plt
import numpy as np
import math
import random


def dist(H, correspondences):
    """ Function, calculates one-way reprojection error

    Parameters
    ----------
    H : np.array (3, 3)
        Homography matrix
    points : np.array (N, 4)
        Correspondences [xs1, ys1, xs2, ys2] for error estimation

    Returns
    -------
    np.array (N, 1)
        Per-correspondence Eucledian squared error
    """

    x1y1 = correspondences[:, :2]
    x2y2 = correspondences[:, 2:]
    
    # Make points affine
    ones = np.ones(len(x1y1))
    A = np.c_[x1y1, ones]

    # Apply projective transformation
    P = A @ H.T

    # Transform to euclidian coordinate system
    x1y1_projected = P[:, :2] / P[:, 2].reshape(-1, 1)

    # Find differences
    diffs = x2y2 - x1y1_projected

    # Find errors
    errors = np.power(diffs, 2).sum(axis=1)
    
    return errors

def zero_one_support(distances, theta):
    inliers = distances <= theta
    support = inliers.sum()
    return support, inliers

def mle_support(distances, theta):
    inliers = distances <= theta
    inliers_distances = distances[inliers]
    support = np.sum(1 - np.power(inliers_distances, 2) / np.power(theta, 2))
    return support, inliers

def get_homography_lstsq(correspondences):
    sample_1 = correspondences[:, :2]
    sample_2 = correspondences[:, 2:]

    T = lambda x, dim: x[:, dim].reshape(-1, 1)

    x, y = T(sample_1, 0), T(sample_1, 1)
    x_prime, y_prime = T(sample_2, 0), T(sample_2, 1)

    ones, zeros = np.ones_like(x), np.zeros_like(x)

    rows_1 = np.concatenate([zeros, zeros, zeros, -x, -y, -ones, y_prime * x, y_prime * y, y_prime], axis=1)
    rows_2 = np.concatenate([-x, -y, -ones, zeros, zeros, zeros, x_prime * x, x_prime * y, x_prime], axis=1)
    last_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.]).reshape(1, -1)

    A = np.concatenate([rows_1, rows_2, last_row], axis=0)
    # A = np.concatenate([rows_1, rows_2], axis=0)

    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.]).reshape(-1, 1)
    # b = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)

    try:
        H = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3,3)
    except:
        print('H does not exist')
        return None
    
    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        print('H[2, 2] == 0')

    return  H

def get_homography_svd(correspondences):
    U1, V1, U2, V2 = correspondences.T

    ones, zeros = np.ones_like(U1), np.zeros_like(U1)

    rows_1 = np.c_[U1, V1, ones, zeros, zeros, zeros, -U2 * U1, -U2 * V1, -U2]
    rows_2 = np.c_[zeros, zeros, zeros, U1, V1, ones, -V2 * U1, -V2 * V1, -V2]

    A = np.r_[rows_1, rows_2]

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3,3)

    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        print('H[2, 2] == 0')

    return  H

 
def ransac(correspondences, n_iters, support_function=mle_support, theta=5, n_samples=4):
    # np.random.seed(45)
    # random.seed(45)

    N = len(correspondences)
    support_best = 0
    H_best = None
    inliers_best = None
    
    for _ in range(n_iters):
        indices = np.random.choice(N, n_samples, replace=False)
        sample = correspondences[indices]
        # H = get_homography_lstsq(sample)
        H = get_homography_svd(sample)

        distances = dist(H, correspondences)

        support, inliers = support_function(distances, theta)
        if support > support_best:
            support_best = support
            H_best = H
            inliers_best = inliers
            print(support)
    
    return H_best, inliers_best


def ransac_2(correspondences, n_iters, H_a, support_function=mle_support, theta=5, n_samples=4):
    # np.random.seed(45)
    # random.seed(45)

    N = len(correspondences)
    support_best = 0
    H_best = None
    a_best = None
    inliers_best = None
    
    for _ in range(n_iters):
        indices = np.random.choice(N, n_samples, replace=False)
        samples = correspondences[indices]
        
        x1, y1, x1_prime, y1_prime = samples[0]
        x2, y2, x2_prime, y2_prime = samples[1]
        x3, y3, x3_prime, y3_prime = samples[2]

        H_a_inv = np.linalg.inv(H_a)

        u1 = np.array([x1, y1, 1])
        u1_prime = np.array([x1_prime, y1_prime, 1])
        u1_prime = H_a_inv @ u1_prime

        u2 = np.array([x2, y2, 1])
        u2_prime = np.array([x2_prime, y2_prime, 1])
        u2_prime = H_a_inv @ u2_prime

        u3 = np.array([x3, y3, 1])
        u3_prime = np.array([x3_prime, y3_prime, 1])
        u3_prime = H_a_inv @ u3_prime
        # print(u3_prime)

        # print(u1)
        x1, y1, w1 = u1
        x2, y2, w2 = u2
        x3, y3, w3 = u3

        x1_prime, y1_prime, w1_prime = u1_prime
        x2_prime, y2_prime, w2_prime = u2_prime
        x3_prime, y3_prime, w3_prime = u3_prime

        v = np.cross(np.cross(u1, u1_prime), np.cross(u2, u2_prime))

        xv, yw, wv = v

        A = np.array([
            (x1_prime * wv - w1_prime * xv) * u1,
            (x2_prime * wv - w2_prime * xv) * u2,
            (x3_prime * wv - w3_prime * xv) * u3,
        ])

        b = np.array([
            x1 * w1_prime - w1 * x1_prime,
            x2 * w2_prime - w2 * x2_prime,
            x3 * w3_prime - w3 * x3_prime,
        ])

        try:
            a = np.linalg.inv(A) @ b
        except:
            print('no inverse')
            continue

        H = np.diag([1, 1, 1]) + np.outer(v, a)

        H_b = H_a @ H
        # H_b = H_b.T

        distances = dist(H_b, correspondences)

        support, inliers = support_function(distances, theta)
        if support > support_best:
            support_best = support
            H_best = H_b
            a_best = a
            inliers_best = inliers
            print(support)
    
    return H_best, inliers_best, a_best


if __name__ == '__main__':
    image = plt.imread('books/book_1.png')
    points_1 = np.loadtxt('books/books_u1.txt')
    X, Y = points_1[:, 0], points_1[:, 1]
    plt.scatter(X, Y, c='k', s=2)

    points_2 = np.loadtxt('books/books_u2.txt')
    X, Y = points_2[:, 0], points_2[:, 1]
    plt.scatter(X, Y, c='r', s=2)

    matches = np.loadtxt('books/books_m12.txt').astype(int)

    correspondences = []
    for i, j in matches:
        # print(i, j)
        x1, y1 = points_1[i]
        x2, y2 = points_2[i]
        correspondences.append([x1, y1, x2, y2])
        # plt.plot([x1, x2], [y1, y2], c='b', alpha=0.3)

    correspondences = np.array(correspondences)

    plt.imshow(image)

    # plt.show()
    # plt.imshow()

    H = np.random.rand(3,3)
    X = np.random.rand(100, 2)
    Y = np.random.rand(100, 2)
    X = np.c_[X, Y]
    H = np.identity(3)
    # print(H)


    distances = dist(H, X)

    H = ransac(correspondences, 10000, theta=25)

    np.set_printoptions(suppress=True)
    print(H)

    # for row in H:
    #     print(','.join([f'{r:.4f}' for r in row]))