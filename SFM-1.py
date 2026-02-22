#This is a baselione code- output is very sparse

#Use more image pairs to densify reconstruction (multi-view SfM).
#Try ORB for more keypoints if SIFT is sparse.
#Combine with dense stereo (e.g., MiDaS/Depth-Anything) for fusion.

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Step 0: Load the two extracted frames
img1 = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Frame Left")
plt.imshow(img1, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Frame Right")
plt.imshow(img2, cmap='gray')
plt.tight_layout()
plt.show()

# Intrinsic camera parameters
focal_length = 1758.23
cx, cy = 872.36, 552.29
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Step 1: Detect keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img_kp1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kp2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Keypoints Left")
plt.imshow(img_kp1)
plt.subplot(1, 2, 2)
plt.title("Keypoints Right")
plt.imshow(img_kp2)
plt.tight_layout()
plt.show()

# Step 2: Match descriptors using FLANN
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
plt.figure(figsize=(12, 6))
plt.title("Step 2: Good Matches")
plt.imshow(img_matches)
plt.show()

# Step 3: Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Step 4: Estimate essential matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
print("Essential Matrix:\n", E)

# Step 5: Recover camera pose
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t)

# Step 6: Triangulate 3D points
proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = K @ np.hstack((R, t))

pts4D = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
pts3D = pts4D[:3] / pts4D[3]  # Convert from homogeneous to 3D
points_3d = pts3D.T  # Shape (N, 3)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

# Optional: Add colors from left image (for visual appeal)
colors = cv2.cvtColor(cv2.imread('left.png'), cv2.COLOR_BGR2RGB)
colors = colors.astype(np.float32) / 255.0
valid_pts = np.int32(pts1)
valid_pts = np.clip(valid_pts, [0, 0], [colors.shape[1] - 1, colors.shape[0] - 1])
point_colors = np.array([colors[y, x] for x, y in valid_pts])
point_colors = point_colors[:points_3d.shape[0]]  # Match shape
pcd.colors = o3d.utility.Vector3dVector(point_colors)

# Visualize
o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud - Open3D")

# Save if needed
o3d.io.write_point_cloud("point_cloud.ply", pcd)
print("Saved point cloud to point_cloud.ply")
