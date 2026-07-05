# Measurement Methodology & Accuracy Report

This report explains the mathematical derivation of the pixel-to-millimeter conversion, the critical dependency on camera calibration, and the final validation results on the test split.

## 1. Pixel-to-MM Mathematical Derivation

To measure physical dimensions from a 2D image, we must resolve two distortions:
1. **Lens Distortion**: Bends straight lines (radial & tangential).
2. **Perspective Projection**: Causes objects further from the camera or tilted relative to the lens to appear smaller or squashed.

Instead of a simple static pixel-to-mm ratio, we derived a **dynamic 3D plane unprojection pipeline** using the estimated camera pose relative to the reference ChArUco board:

### Projective Plane Homography
Any point on the flat reference plane (where the board and card lie) can be represented in the board's coordinate system as $P_{\text{board}} = (x, y, 0)^T$. 
Using the camera intrinsic matrix $K$, and the estimated camera pose rotation matrix columns $R_1, R_2$ and translation vector $T$, the mapping from the physical plane coordinates to the undistorted image pixel coordinates $(u, v)$ is defined by a $3 \times 3$ homography matrix $H$:
$$
\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \sim H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix} \quad \text{where} \quad H = K \begin{pmatrix} R_1 & R_2 & T \end{pmatrix}
$$

### 3D Unprojection
By computing the inverse homography $H^{-1}$, we can project any pixel $(u, v)$ in the undistorted image back onto the physical board plane $(x, y)$ in meters:
$$
\begin{pmatrix} x \\ y \\ 1 \end{pmatrix} \sim H^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
$$

For every pixel belonging to the predicted card mask, we apply this unprojection. The resulting collection of points represents the card's physical shape in the 2D plane of the board.

### Metric Bounding
We fit a minimum area bounding box `cv2.minAreaRect` around the unprojected coordinates to find the physical length and width (in meters, converted to millimeters). This handles any random placement or rotation of the card.

---

## 2. Calibration Dependency

Raw (distorted) images produce incorrect metric measurements for two reasons:
1. **Lens Distortion**: Radial distortion bends lines (barrel or pincushion effect). A straight card edge will appear curved in pixels, distorting length calculations.
2. **Perspective Scale Skew**: A card placed at the edge of the frame or at a tilted angle undergoes perspective compression. Without pose estimation ($R, T$) to map the tilt angle and distance, a simple 2D pixel ratio will underestimate the dimensions.

Undistorting the image with $K$ and $D$, and unprojecting with $H^{-1}$ solves both issues.

---

## 3. Accuracy Validation Table

The pipeline was validated on $12$ held-out test images containing a playing card (**$88.90 \text{ mm} \times 63.50 \text{ mm}$**):

| Image File | Card Type | Measured Width | Measured Height | Width Error | Height Error | Width % Error | Height % Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sample_0000.png | heart | 91.40 mm | 65.80 mm | +2.50 mm | +2.30 mm | 2.81% | 3.62% |
| sample_0001.png | heart | 91.01 mm | 65.20 mm | +2.11 mm | +1.70 mm | 2.38% | 2.67% |
| sample_0002.png | spade | 92.21 mm | 65.70 mm | +3.31 mm | +2.20 mm | 3.72% | 3.46% |
| sample_0003.png | spade | 90.44 mm | 65.80 mm | +1.54 mm | +2.30 mm | 1.73% | 3.62% |
| sample_0004.png | spade | 90.78 mm | 65.75 mm | +1.88 mm | +2.25 mm | 2.12% | 3.54% |
| sample_0005.png | heart | 90.89 mm | 66.16 mm | +1.99 mm | +2.66 mm | 2.23% | 4.18% |
| sample_0006.png | heart | 91.08 mm | 65.69 mm | +2.18 mm | +2.19 mm | 2.45% | 3.45% |
| sample_0007.png | heart | 90.73 mm | 65.91 mm | +1.83 mm | +2.41 mm | 2.06% | 3.80% |
| sample_0008.png | heart | 92.14 mm | 66.24 mm | +3.24 mm | +2.74 mm | 3.65% | 4.32% |
| sample_0009.png | heart | 91.36 mm | 65.97 mm | +2.46 mm | +2.47 mm | 2.77% | 3.89% |
| sample_0010.png | heart | 91.01 mm | 66.39 mm | +2.11 mm | +2.89 mm | 2.37% | 4.55% |
| sample_0011.png | spade | 91.40 mm | 66.19 mm | +2.50 mm | +2.69 mm | 2.81% | 4.24% |

### Global Accuracy Summary
- **Mean Absolute Error (MAE) - Width**: **$2.3036 \text{ mm}$**
- **Mean Absolute Error (MAE) - Height**: **$2.3995 \text{ mm}$**
- **Mean Percentage Error (MPE) - Width**: **$2.5913\%$**
- **Mean Percentage Error (MPE) - Height**: **$3.7788\%$**
- **Average Overall Error**: **$3.1850\%$**

---

## 4. Assumptions & Limitations
- **Coplanar Constraint**: The target object must lie flat on the same physical plane as the reference ChArUco board. If the object has significant 3D height, the top surface will appear larger due to perspective compression (closer to the lens).
- **Reference Dependency**: The reference ChArUco board must be visible in the image. If the board is fully covered or out of view, pose estimation ($H^{-1}$) cannot be computed, and the pipeline falls back to generic scale estimates.
