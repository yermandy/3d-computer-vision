# %%
import matplotlib.pyplot as plt
import numpy as np


def get_line(x1, y1, x2, y2, plot=False, c='k'):
    if plot:
        plt.scatter([x1, x2], [y1, y2], c=c)
    return np.cross([x1, y1, 1], [x2, y2, 1])


def plot_line(x1, y1, x2, y2, c='k'):
    plt.plot([x1, x2], [y1, y2], c=c)


def lines_intersection(l1, l2):
    x = np.cross(l1, l2)
    x = [x[0] / x[2], x[1] / x[2]]
    return x


def inside_area(x1, y1, x2, y2, x, y) :
    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return True
    else:
        return False


def plot_line_within_box(l1, box, c='k'):
    intersections = []

    for l2 in box:

        x = lines_intersection(l1, l2)

        if inside_area(0, 0, width, height, *x):
            intersections.append(x)
            # plt.scatter(*x, c=c)

    return intersections
    # print(intersections)
    # plot_line(*intersections, c=c)


def create_point(x, y):
    return np.array([x, y])


def e2p(x):
    return np.array([*x, 1])


def p2e(x):
    return [x[0] / x[2], x[1] / x[2]]


# %%

width, height = 800, 600

fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1])

plt.gca().invert_yaxis()

x = plt.ginput(4)

x = np.array(x) 

x[:, 0] *= width
x[:, 1] *= height

point_1 = x[0]
point_2 = x[1]
point_3 = x[2]
point_4 = x[3]

plt.close()


# %%

plt.gca().invert_yaxis()
plt.axis('equal')

width, height = 800, 600

top_left = create_point(0, 0)
top_right = create_point(width, 0)
bottom_left = create_point(0, height)
bottom_right = create_point(width, height)

top_left_bottom_left = get_line(*top_left, *bottom_left)
top_left_top_right = get_line(*top_left, *top_right)
top_right_bottom_right = get_line(*top_right, *bottom_right)
bottom_left_bottom_right = get_line(*bottom_left, *bottom_right)

plot_line(*top_left, *bottom_left)
plot_line(*top_left, *top_right)
plot_line(*top_right, *bottom_right)
plot_line(*bottom_left, *bottom_right)

box = [
    top_left_bottom_left,
    top_left_top_right,
    top_right_bottom_right,
    bottom_left_bottom_right
]

print(point_1)
print(point_2)
print(point_3)
print(point_4)
# point_1 = [100, 300]
# point_2 = [200, 400]
# point_3 = [600, 300]
# point_4 = [700, 200]

l1 = get_line(*point_1, *point_2, True, 'b')
intersections_1 = plot_line_within_box(l1, box, 'b')
plot_line(*intersections_1[0], *intersections_1[1], c='b')

l2 = get_line(*point_3, *point_4, True, 'g')
intersections_2 = plot_line_within_box(l2, box, 'g')
plot_line(*intersections_2[0], *intersections_2[1], c='g')

intersection = lines_intersection(l1, l2)
plt.scatter(*intersection, c='r', linewidths=5);

# %%

# plt.gca().invert_yaxis()
plt.axis('equal')

K = np.array(
    [[    1,   0.1, 0 ], 
    [   0.1,     1, 0 ],
    [ 0.004, 0.002, 1 ]])

top_left = K @ e2p(top_left)
bottom_left = K @ e2p(bottom_left)
top_right = K @ e2p(top_right)
bottom_right = K @ e2p(bottom_right)

top_left = p2e(top_left)
bottom_left = p2e(bottom_left)
top_right = p2e(top_right)
bottom_right = p2e(bottom_right)

top_left_bottom_left = K @ top_left_bottom_left
top_left_top_right = K @ top_left_top_right
top_right_bottom_right = K @ top_right_bottom_right
bottom_left_bottom_right = K @ bottom_left_bottom_right

plot_line(*top_left, *bottom_left)
plot_line(*top_left, *top_right)
plot_line(*top_right, *bottom_right)
plot_line(*bottom_left, *bottom_right)

point_1 = K @ e2p(point_1)
point_2 = K @ e2p(point_2)
point_3 = K @ e2p(point_3)
point_4 = K @ e2p(point_4)

point_1 = p2e(point_1)
plt.scatter(*point_1, c='b')

point_2 = p2e(point_2)
plt.scatter(*point_2, c='b')

point_3 = p2e(point_3)
plt.scatter(*point_3, c='g')

point_4 = p2e(point_4)
plt.scatter(*point_4, c='g')

i1_1 = K @ e2p(intersections_1[0])
i1_1 = p2e(i1_1)

i2_1 = K @ e2p(intersections_1[1])
i2_1 = p2e(i2_1)

i1_2 = K @ e2p(intersections_2[0])
i1_2 = p2e(i1_2)

i2_2 = K @ e2p(intersections_2[1])
i2_2 = p2e(i2_2)

plot_line(*i1_1, *i2_1, c='b')
plot_line(*i1_2, *i2_2, c='g')

intersection = K @ e2p(intersection)
intersection = p2e(intersection)

plt.scatter(*intersection, c='r', linewidths=5);

plt.show()
# %%
