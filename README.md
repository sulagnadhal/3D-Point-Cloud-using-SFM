# Structure From Motion (SfM) – Stereo 3D Reconstruction

## Overview
This project implements a basic Structure From Motion (SfM) / Stereo Reconstruction pipeline using OpenCV.

It takes a pair of stereo images (left and right views), detects matching feature points, estimates camera motion, and reconstructs 3D structure from the image pair.

This project demonstrates core computer vision concepts including:
- Feature detection & matching
- Essential matrix estimation
- Camera pose recovery
- Triangulation
- 3D point reconstruction

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

##  Project Structure
├── SFM-1.py # Feature detection and matching
├── SFM-2.py # Pose estimation and triangulation
├── left.png # Left stereo image
├── right.png # Right stereo image
└── README.md


---

##  How It Works

### Load Stereo Images
The program loads:
- `left.png`
- `right.png`
  
<p float="left">
  <img src="left.png" width="45%" />
  <img src="right.png" width="45%" />
</p>

###  Feature Detection
ORB/SIFT features are detected in both images.

###  Feature Matching
Matching keypoints are found using:
- BFMatcher or FLANN

###  Essential Matrix Estimation
Using matched points:

cv2.findEssentialMat()

### Recover Camera Pose
cv2.recoverPose()
This estimates:
Rotation (R)
Translation (t)

### Triangulation

3D points are reconstructed using:
cv2.triangulatePoints()

## Output 3D Image
<img src="output1.png" width="400">

## How to Run
Step 1: Install Dependencies

pip install opencv-python numpy matplotlib

Step 2: Run the Script
python SFM-1.py

Author
Sulagna Dhal – Computer Science
Computer Vision & 3D Reconstruction
