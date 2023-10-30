#Library 
import cv2 
import os 
import numpy as np
import matplotlib.pyplot as plt

#List & Load
image_scene = cv2.imread('./kitkat_scene.jpg')
image_TargetList = []
image_PathList = []

for path in os.listdir('./targetList'):
    image_PathList.append('./targetList/' + path)
print(image_PathList)

#Smoothing 
for i in range (len(image_PathList)):
    image = cv2.imread(image_PathList[i])
    image = cv2.blur(image, (2,2))
    ret, image = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
    image_TargetList.append(image)

#Feature Detection 
SIFT = cv2.SIFT_create()
AKAZE = cv2.AKAZE_create()
ORB = cv2.ORB_create()

sift_keypoint_target = []
sift_descriptor_target = []
sift_keypoint_scene, sift_descriptor_scene = SIFT.detectAndCompute(image_scene, None)
sift_descriptor_scene = np.float32(sift_descriptor_scene)

akaze_keypoint_target = []
akaze_descriptor_target = []
akaze_keypoint_scene, akaze_descriptor_scene = AKAZE.detectAndCompute(image_scene, None)
akaze_descriptor_scene = np.float32(akaze_descriptor_scene)

orb_keypoint_target = []
orb_descriptor_target = []
orb_keypoint_scene, orb_descriptor_scene = ORB.detectAndCompute(image_scene, None)

for image in image_TargetList:
    s_keypoint, s_descriptor = SIFT.detectAndCompute(image, None)
    s_descriptor = np.float32(s_descriptor)
    sift_keypoint_target.append(s_keypoint)
    sift_descriptor_target.append(s_descriptor)

    a_keypoint, a_descriptor = AKAZE.detectAndCompute(image, None)
    a_descriptor = np.float32(a_descriptor)
    akaze_keypoint_target.append(a_keypoint)
    akaze_descriptor_target.append(a_descriptor)

    o_keypoint, o_descriptor = ORB.detectAndCompute(image, None)
    orb_keypoint_target.append(o_keypoint)
    orb_descriptor_target.append(o_descriptor)

#Feature Matching 
flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(check=100))
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def createMasking(mask, match):
    for i, (fm, sm) in enumerate(match):
        if fm.distance < 0.7 * sm.distance:
            mask[i] = [1,0]
    return mask

sift_match_result = []
for s_descriptor in sift_descriptor_target:
    sift_match_result.append(flann.knnMatch(s_descriptor, sift_descriptor_scene, 2))

sift_result = max(sift_match_result, key=len)
sift_image_index = sift_match_result.index(sift_result)
sift_matches_mask = [[0,0] for i in range (len(sift_result))]
sift_matches_mask = createMasking(sift_matches_mask, sift_result)

sift_res = cv2.drawMatchesKnn(
    image_TargetList[sift_image_index],
    sift_keypoint_target[sift_image_index],
    image_scene, sift_keypoint_scene,
    sift_result, None,
    matchColor=[255,0,0],
    singlePointColor=[0,255,0],
    matchesMask=sift_matches_mask
)
sift_res = cv2.cvtColor(sift_res, cv2.COLOR_BGR2RGB)

akaze_match_result = []
for a_descriptor in akaze_descriptor_target:
    akaze_match_result.append(flann.knnMatch(a_descriptor, akaze_descriptor_scene, 2))

akaze_result = max(akaze_match_result, key=len)
akaze_image_index = akaze_match_result.index(akaze_result)
akaze_matches_mask = [[0,0] for i in range (len(akaze_result))]
akaze_matches_mask = createMasking(akaze_matches_mask, akaze_result)

akaze_res = cv2.drawMatchesKnn(
    image_TargetList[akaze_image_index],
    akaze_keypoint_target[akaze_image_index],
    image_scene, akaze_keypoint_scene,
    akaze_result, None,
    matchColor=[255,0,0],
    singlePointColor=[0,255,0],
    matchesMask=akaze_matches_mask
)
akaze_res = cv2.cvtColor(akaze_res, cv2.COLOR_BGR2RGB)

orb_match_result = []
for o_descriptor in orb_descriptor_target:
    match_result = bfmatcher.match(o_descriptor, orb_descriptor_scene)
    match_result = sorted(match_result, key=lambda x : x.distance)
    orb_match_result.append(match_result)

orb_result = max(orb_match_result, key=len)
orb_image_index = orb_match_result.index(orb_result)

orb_res = cv2.drawMatches(
    image_TargetList[orb_image_index],
    orb_keypoint_target[orb_image_index],
    image_scene, orb_keypoint_scene,
    orb_result[:50], None,
    matchColor=[255,0,0],
    singlePointColor=[0,255,0],
    flags=2
)
orb_res = cv2.cvtColor(orb_res, cv2.COLOR_BGR2RGB)

#Result
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.imshow(sift_res, cmap='gray')
plt.title('SIFT')

plt.subplot(2,2,2)
plt.imshow(akaze_res, cmap='gray')
plt.title('AKAZE')

plt.subplot(2,2,3)
plt.imshow(orb_res, cmap='gray')
plt.title('ORB')

plt.show()