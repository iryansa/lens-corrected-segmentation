import os
import json
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

def load_camera_params(path):
    with open(path, "r") as f:
        params = json.load(f)
    camera_matrix = np.array(params["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(params["distortion_coefficients"], dtype=np.float64)
    return camera_matrix, dist_coeffs

def estimate_pose_and_get_homography(img_gray, detector, board, camera_matrix, dist_coeffs):
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(img_gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None, None, None, None

    # Solve PnP to get camera pose relative to the board plane (z=0)
    chessboard_corners = board.getChessboardCorners()
    obj_points = chessboard_corners[charuco_ids].reshape(-1, 3)
    image_points = charuco_corners.reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs
    )
    if not success:
        return None, None, None, None

    # Compute homography H mapping from board plane (z=0) to image plane (pixels)
    R, _ = cv2.Rodrigues(rvec)
    # H = K * [R1, R2, T]
    H = camera_matrix @ np.column_stack((R[:, 0], R[:, 1], tvec))
    H_inv = np.linalg.inv(H)

    return H_inv, rvec, tvec, charuco_corners

def main():
    parser = argparse.ArgumentParser(description="Measure physical object dimensions in mm from a calibrated image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input raw image.")
    parser.add_argument("--output-dir", type=str, default="measurements/outputs", help="Directory to save outputs.")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Path to model weights.")
    parser.add_argument("--camera-params", type=str, default="calibration/camera_params.json", help="Path to camera parameters JSON.")
    args = parser.parse_args()

    # Ground truth dimensions of the playing card
    GT_WIDTH_MM = 88.90
    GT_HEIGHT_MM = 63.50

    # 1. Load Camera Calibration Parameters
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)

    # 2. Setup ChArUco Board for Reference Calibration
    squares_x = 8
    squares_y = 10
    square_length = 0.024  # square size in meters
    marker_length = 0.018  # marker size in meters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_length, 
        marker_length, 
        aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(board)

    # 3. Read & Undistort Image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at {args.image}")
        return

    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
    img_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img_undistorted.shape[:2]

    # 4. Detect ChArUco Board Reference & Get Inverse Homography (Plane Unprojection)
    H_inv, rvec, tvec, charuco_corners = estimate_pose_and_get_homography(
        img_gray, 
        detector, 
        board, 
        camera_matrix, 
        dist_coeffs
    )

    if H_inv is None:
        print("Warning: ChArUco board reference could not be detected/resolved.")
        print("Fallback mode: Using calibration file distance ratio (might be inaccurate due to depth/perspective).")
        # Fallback ratio: We'll calculate a generic pixels_per_mm if pose estimation fails
        # But for our pipeline, we assume board detection is active.
        return

    # 5. Load Model & Predict Segmentation Mask
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    img_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs)
        mask_resized = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255

    pred_mask = cv2.resize(mask_resized, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # 6. Extract Mask Contours & Check If Object Exists
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Error: No target object segmented in the image!")
        return

    # Keep the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:  # noise filter
        print("Error: Segmented target area is too small (likely noise)!")
        return

    # Get all pixel coordinates inside the contour
    mask_indices = np.where(pred_mask > 0)
    # Convert to (x, y) coordinates
    pixel_pts = np.column_stack((mask_indices[1], mask_indices[0]))

    # 7. Unproject pixel coordinates onto the flat board plane (z = 0)
    N = len(pixel_pts)
    pts_hom = np.column_stack((pixel_pts, np.ones(N)))
    pts_board_hom = pts_hom @ H_inv.T
    pts_board = pts_board_hom[:, :2] / pts_board_hom[:, 2:3]  # in meters

    # 8. Compute Minimum Area Bounding Box in the physical board plane
    rect = cv2.minAreaRect(pts_board.astype(np.float32))
    (cx_m, cy_m), (dim1_m, dim2_m), angle = rect

    # Convert physical dimensions to millimeters
    measured_w = max(dim1_m, dim2_m) * 1000.0
    measured_h = min(dim1_m, dim2_m) * 1000.0

    # Calculate errors
    err_w = measured_w - GT_WIDTH_MM
    err_h = measured_h - GT_HEIGHT_MM
    pct_err_w = (abs(err_w) / GT_WIDTH_MM) * 100.0
    pct_err_h = (abs(err_h) / GT_HEIGHT_MM) * 100.0

    print("-" * 50)
    print("Measurement Results:")
    print(f"  Width:  {measured_w:.2f} mm (Ground Truth: {GT_WIDTH_MM:.2f} mm | Error: {err_w:+.2f} mm, {pct_err_w:.2f}%)")
    print(f"  Height: {measured_h:.2f} mm (Ground Truth: {GT_HEIGHT_MM:.2f} mm | Error: {err_h:+.2f} mm, {pct_err_h:.2f}%)")
    print("-" * 50)

    # 9. Visualization & Drawing
    # We can project the 4 corners of the fitted physical bounding box back to image pixels for overlay drawing
    box_pts_board = cv2.boxPoints(rect)  # (4, 2) in meters
    box_pts_board_3d = np.column_stack((box_pts_board, np.zeros(4)))  # add z=0
    
    img_box_pts, _ = cv2.projectPoints(
        box_pts_board_3d.reshape(-1, 1, 3), 
        rvec, 
        tvec, 
        camera_matrix, 
        dist_coeffs
    )
    img_box_pts = img_box_pts.reshape(-1, 2).astype(np.int32)

    # Create overlay drawing
    annotated = img_undistorted.copy()
    
    # Draw ChArUco corners
    cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners)

    # Draw translucent mask
    overlay = annotated.copy()
    overlay[pred_mask > 0] = [0, 255, 0]
    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    # Draw physical 3D-aligned bounding box (in red)
    cv2.drawContours(annotated, [img_box_pts], -1, (0, 0, 255), 8)

    # Write dimension text on the image
    box_center = np.mean(img_box_pts, axis=0).astype(np.int32)
    # Add measurements text
    cv2.putText(annotated, f"W: {measured_w:.1f} mm", (box_center[0] - 150, box_center[1] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
    cv2.putText(annotated, f"H: {measured_h:.1f} mm", (box_center[0] - 150, box_center[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    out_img_path = os.path.join(args.output_dir, os.path.basename(args.image))
    cv2.imwrite(out_img_path, annotated)
    print(f"Annotated measurement image saved to: {out_img_path}")

    # Return results for accuracy validation script
    return {
        "image": os.path.basename(args.image),
        "measured_w": measured_w,
        "measured_h": measured_h,
        "err_w": err_w,
        "err_h": err_h,
        "pct_err_w": pct_err_w,
        "pct_err_h": pct_err_h
    }

if __name__ == "__main__":
    main()
