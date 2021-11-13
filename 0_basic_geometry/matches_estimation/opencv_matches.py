# %%

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Images are taken from HSequences dataset https://github.com/hpatches/hpatches-dataset
img1 = cv2.cvtColor(cv2.imread('books/book_1.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('books/book_2.png'), cv2.COLOR_BGR2RGB)
# H_gt = np.loadtxt('v_woman_H_1_6')


def show_two_images(img1, img2):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    return


# %%
det = cv2.AKAZE_create()

kps1, descs1 = det.detectAndCompute(img1, None)
kps2, descs2 = det.detectAndCompute(img2, None)


vis_img1, vis_img2 = None, None
vis_img1 = cv2.drawKeypoints(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), kps1, vis_img1,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
vis_img2 = cv2.drawKeypoints(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), kps2, vis_img2,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_two_images(vis_img1, vis_img2)

# %%

def match_descriptors(descriptors1, descriptors2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    ratio_thresh = 0.8
    tentative_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            tentative_matches.append(m)
    return tentative_matches

def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


tentative_matches = match_descriptors(descs1, descs2)
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)

cv2.drawMatches(decolorize(img1), kps1, decolorize(img2), kps2, tentative_matches, img_matches, 
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(20,10))
plt.imshow(img_matches)

# %%
for kp in kps1:
    print(kp)
# %%

plt.imshow(img_matches)
cv2.DMatch
# %%
keypoints_1 = []
for kp in kps1:
    keypoints_1.append(kp.pt)
# keypoints_1 = np.round(keypoints_1).astype(int)
keypoints_1 = np.round(keypoints_1, 2)
np.savetxt('keypoints_1.txt', keypoints_1, fmt='%s')

keypoints_2 = []
for kp in kps2:
    keypoints_2.append(kp.pt)
# keypoints_2 = np.round(keypoints_2).astype(int)
keypoints_2 = np.round(keypoints_2, 2)
np.savetxt('keypoints_2.txt', keypoints_2, fmt='%s')

# %%
MATCHES = []
for tm in tentative_matches:
    print(tm.queryIdx, tm.trainIdx)
    MATCHES.append([tm.queryIdx, tm.trainIdx])

MATCHES = np.array(MATCHES).astype(int)
np.savetxt('matches_12.txt', MATCHES, fmt='%s')
# %%
