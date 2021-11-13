import matplotlib.pyplot as plt
import numpy as np
import math

def points_in_homogeneous(cartesian_points):
    return np.c_[cartesian_points, np.ones(len(cartesian_points))]
    
def line_from_points(x1, y1, x2, y2):
    return np.cross([x1, y1, 1], [x2, y2, 1])

def line_in_homogeneous(cartesian_line):
    # cartesian line in form of y = ax + b, i.e. [a, b] vector
    a, b = cartesian_line
    # find (x1, y1) point
    x1, y1 = 0, b
    # find (x2, y2) point
    x2, y2 = -b/a, 0
    # find line in homogeneous system
    # https://math.stackexchange.com/questions/773904/cartesian-line-to-projective-coordinates
    # line = line_from_points(x1, y1, x2, y2)
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])
    return line

def lines_intersection(l1, l2):
    x = np.cross(l1, l2)
    x = [x[0] / x[2], x[1] / x[2]]
    return x

def find_distances(points, line):
    # cast points from cartesian to homogeneous
    points = points_in_homogeneous(points)
    distances = points @ line / math.sqrt(line[0] ** 2 + line[1] ** 2)
    distances = np.abs(distances)
    return distances

def lstsq(U, V):
    line = np.linalg.lstsq(np.c_[U, np.ones_like(U)], V, rcond=None)[0]
    line = line_in_homogeneous(line)
    return line

def zero_one_support(distances, theta):
    inliers = distances <= theta
    support = inliers.sum()
    return support, inliers

def mle_support(distances, theta):
    inliers = distances <= theta
    inliers_distances = distances[inliers]
    support = np.sum(1 - np.power(inliers_distances, 2) / np.power(theta, 2))
    return support, inliers

def ransac(points, n_iters, support_function=zero_one_support, theta=5, with_lstsq=False):
    np.random.seed(45)

    n_points = len(points)
    support_best = 0
    line_best = None
    inliers_best = None
    
    for _ in range(n_iters):
        indices = np.random.choice(n_points, 2, replace=False)
        x1, y1, x2, y2 = points[indices].flatten()
        line = line_from_points(x1, y1, x2, y2)
        distances = find_distances(points, line)
        support, inliers = support_function(distances, theta)
        if support > support_best:
            support_best = support
            line_best = line
            inliers_best = inliers
    
    if with_lstsq:
        support_points = points[inliers_best]
        U, V = support_points[:, 0], support_points[:, 1]
        line_best = lstsq(U, V)

    return line_best

def plot_line(line, u_min, u_max, v_min, v_max, label=''):
    l1 = line_from_points(u_min, v_min, u_min, v_max)
    l2 = line_from_points(u_min, v_min, u_max, v_min)
    l3 = line_from_points(u_max, v_max, u_max, v_min)
    l4 = line_from_points(u_max, v_max, u_min, v_max)

    u1, v1 = lines_intersection(l1, line)
    u2, v2 = lines_intersection(l2, line)
    u3, v3 = lines_intersection(l3, line)
    u4, v4 = lines_intersection(l4, line)

    plt.plot([u1, u2, u3, u4], [v1, v2, v3, v4], label=label)


data = np.array([[0, 0], [1, 1], [2, 1], [3, 2], [4, 2], [5, 3]])
data = np.loadtxt('points/linefit_3.txt')

U, V = data[:, 0], data[:, 1]

## Find borders
u_min, u_max = np.min(U), np.max(U)
v_min, v_max = np.min(V), np.max(V)

## Set borders
plt.xlim(u_min, u_max)
plt.ylim(v_min, v_max)
# plt.axis('equal')
# plt.axis('square')

## Plot points
plt.scatter(U, V, s=5, c='k')

## Least Squares
line = lstsq(U, V)
plot_line(line, u_min, u_max, v_min, v_max, label='lstsq')

n_iters = 100

## RANSAC
ransac_line = ransac(data, n_iters)
plot_line(ransac_line, u_min, u_max, v_min, v_max, label='ransac')

## RANSAC with Least Squares
ransac_lstsq_line = ransac(data, n_iters, with_lstsq=True)
plot_line(ransac_lstsq_line, u_min, u_max, v_min, v_max, label='ransac + lstsq')

## MLESAC with Least Squares
mlesac_lstsq_line = ransac(data, n_iters, support_function=mle_support, with_lstsq=True)
plot_line(mlesac_lstsq_line, u_min, u_max, v_min, v_max, label='mlesac + lstsq')

## Original line
original_line = [-10, 3, 1200]
plot_line(original_line, u_min, u_max, v_min, v_max, label='original')


def normalize_line(line):
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
    norm = math.sqrt(line[0] ** 2 + line[1] ** 2)
    line = np.array(line) / norm
    return line

def angle_between_lines(line_1, line_2):
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
    line_1 = normalize_line(line_1)[:2]
    line_2 = normalize_line(line_2)[:2]
    return math.degrees(math.acos(line_1 @ line_2))

def dist_to_origin(line):
    # https://math.stackexchange.com/questions/1958856/equation-for-distance-of-the-straight-line-from-the-origin
    return abs(line[2]) / math.sqrt(line[0] ** 2 + line[1] ** 2)

x_axis_line = line_from_points(0, 0, 1, 0)

angle_1 = angle_between_lines(ransac_line, x_axis_line)
angle_2 = angle_between_lines(ransac_lstsq_line, x_axis_line)
angle_3 = angle_between_lines(mlesac_lstsq_line, x_axis_line)
angle_4 = angle_between_lines(original_line, x_axis_line)

intersection_1 = lines_intersection(ransac_line, x_axis_line)
intersection_2 = lines_intersection(ransac_lstsq_line, x_axis_line)
intersection_3 = lines_intersection(mlesac_lstsq_line, x_axis_line)
intersection_4 = lines_intersection(original_line, x_axis_line)

dist_1 = dist_to_origin(ransac_line)
dist_2 = dist_to_origin(ransac_lstsq_line)
dist_3 = dist_to_origin(mlesac_lstsq_line)
dist_4 = dist_to_origin(original_line)

print(f'ransac       angle {angle_1:.4f} | distance {dist_1:.4f}')
print(f'ransac lsqst angle {angle_2:.4f} | distance {dist_2:.4f}')
print(f'mlesac lsqst angle {angle_3:.4f} | distance {dist_3:.4f}')
print(f'original     angle {angle_4:.4f} | distance {dist_4:.4f}')

plt.tight_layout()
plt.legend()
plt.show()

# See also
# https://math.stackexchange.com/questions/3300524/parallel-and-perpendicular-line-through-a-point