from lib import *
from homographies import ransac, ransac_2

def plot_grid(H, height, width):
    points = []
    for x in np.arange(0, width, 15):
        for y in np.arange(0, height, 15):
            points.append([x, y])
    points = np.array(points)

    points = np.c_[points, np.ones(len(points))]
    points = points @ H.T
    points = points[:, :2] / points[:, 2].reshape(-1, 1)

    plt.scatter(points[:, 0], points[:, 1], alpha=0.5, marker='.')


def plot_line(line, width, height):

    l1 = line_from_points(0, 0, width, 0)
    l2 = line_from_points(0, height, width, height)

    x1, y1 = lines_intersection(line, l1)
    x2, y2 = lines_intersection(line, l2)

    plt.plot([x1, x2], [y1, y2], c='magenta')


def plot_needle_map(correspondences, c='r'):
    for x1, y1, x2, y2 in correspondences:
        plt.plot([x1, x2], [y1, y2], c=c, linewidth=1)
    plt.scatter(correspondences[:, 0], correspondences[:, 1], marker='o', s=5, c=c)


def show(width, height):
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    book_1_id = 1
    book_2_id = 3

    image_1 = plt.imread(f'books/book_{book_1_id}.png')
    # plt.imshow(image_1, alpha=0.5)
    plt.imshow(image_1, alpha=1)

    image_2 = plt.imread(f'books/book_{book_2_id}.png')
    # plt.imshow(image_2, alpha=0.5)

    height, width = image_1.shape[0], image_1.shape[1]

    points_1 = np.loadtxt(f'books/books_u{book_1_id}.txt')
    # points_1 = np.loadtxt('matches_estimation/keypoints_1.txt')
    X, Y = points_1[:, 0], points_1[:, 1]
    plt.scatter(X, Y, c='r', s=2)

    points_2 = np.loadtxt(f'books/books_u{book_2_id}.txt')
    # points_2 = np.loadtxt('matches_estimation/keypoints_2.txt')
    X, Y = points_2[:, 0], points_2[:, 1]
    plt.scatter(X, Y, c='k', s=2)


    matches = np.loadtxt(f'books/books_m{book_1_id}{book_2_id}.txt').astype(int)
    # matches = np.loadtxt('matches_estimation/matches_12.txt').astype(int)

    correspondences = []
    for i, j in matches:
        x1, y1 = points_1[i]
        x2, y2 = points_2[j]
        correspondences.append([x1, y1, x2, y2])
    correspondences = np.array(correspondences)

    H_a, inliers = ransac(correspondences, 10000, theta=4)
    # plot_grid(H_a, height, width)
    plot_needle_map(correspondences[inliers], c='tab:red')


    correspondences = correspondences[~inliers]

    H_b, inliers, a_best = ransac_2(correspondences, 10000, H_a, theta=3, n_samples=3)
    # plot_grid(H_a, height, width)
    plot_needle_map(correspondences[inliers], c='tab:green')


    plot_line(a_best, width, height)

    show(width, height)