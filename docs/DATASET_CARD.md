# Dataset Card: Playing Card Dataset

This document details the dataset used to train the deep learning image segmentation model.

## 1. Object Description
The target object chosen for the measurement technical evaluation is a **Standard Playing Card** (specifically standard Poker card size).
- **Physical Bounding Size**: **$88.90 \text{ mm} \times 63.50 \text{ mm}$** ($3.5\text{"} \times 2.5\text{"}$).
- **Geometry**: Planar rectangle with rounded corners (radius of ~$3.18\text{ mm}$).
- **Rationale**: Planar rectangles are ideal for metric measurements as they lie flat on the reference plane, minimizing 3D perspective depth offsets. Playing cards are universally available, have distinct high-contrast borders and suits, and standard dimensions.

## 2. Collection & Labelling Strategy
To comply with the headless execution constraints, a **synthetic dataset generation** pipeline was engineered in [generate_dataset.py](file:///e:/xis.ai/dataset/generate_dataset.py). 

Instead of manual manual annotation (which is prone to pixel-level boundary errors), we implemented a **3D-to-2D projective synthesis pipeline**:
1. **Pose Detection**: For each calibration image, the ChArUco board reference pattern was detected, and the 3D camera pose (rotation $R$, translation $T$) was resolved using PnP solver with the calibrated camera matrix.
2. **Dynamic Placement**: The playing card's 3D coordinates (flat on $z=0$) were randomized on the board plane.
3. **Perspective Projection**: The card's 3D boundary was projected back to 2D pixel coordinates using `cv2.projectPoints` and the camera intrinsics/distortion parameters.
4. **Warping & Blending**: High-resolution playing card templates (Ace of Spades, King of Hearts) were warped into the projected coordinates and alpha-blended with Gaussian edge smoothing and brightness variations.
5. **Mask Generation**: The exact warped region served as the pixel-perfect ground-truth mask.

This approach guarantees zero-noise annotation boundaries, realistic perspective distortion, and identical lens distortion.

## 3. Dataset Statistics
The dataset is split as follows:
- **Total Samples**: $116$ images.
- **Train Set**: $81$ images ($70\%$).
- **Validation Set**: $23$ images ($20\%$).
- **Test Set**: $12$ images ($10\%$).
- **Image Resolution**: $2992 \times 3992$ pixels ($12 \text{ Megapixels}$).
- **Classes**: 1 class (`playing_card`).
- **Suit Distribution**: ACE of Spades (~$50\%$), KING of Hearts (~$50\%$).
- **Metadata**: Each split contains a `metadata.json` documenting the template used, physical placement ($x, y$ coordinates in meters), rotation angle (radians), and projected corner pixels.
