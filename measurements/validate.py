import os
import json
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

    R, _ = cv2.Rodrigues(rvec)
    H = camera_matrix @ np.column_stack((R[:, 0], R[:, 1], tvec))
    H_inv = np.linalg.inv(H)

    return H_inv, rvec, tvec, charuco_corners

def main():
    GT_WIDTH_MM = 88.90
    GT_HEIGHT_MM = 63.50

    camera_params_path = "calibration/camera_params.json"
    weights_path = "models/best_model.pth"
    test_dir = "dataset/test"
    output_dir = "measurements/outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading camera parameters and U-Net model...")
    camera_matrix, dist_coeffs = load_camera_params(camera_params_path)

    squares_x = 8
    squares_y = 10
    square_length = 0.024
    marker_length = 0.018
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_length, 
        marker_length, 
        aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(board)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test metadata to map card type
    with open(os.path.join(test_dir, "metadata.json"), "r") as f:
        test_metadata = json.load(f)

    results = []
    print(f"Starting measurement validation on {len(test_metadata)} test images...")

    for i, meta in enumerate(test_metadata):
        img_name = meta["image_file"]
        img_path = os.path.join(test_dir, "images", img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Undistort and preprocess
        img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        img_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = img_undistorted.shape[:2]

        # Pose and Homography
        H_inv, rvec, tvec, charuco_corners = estimate_pose_and_get_homography(
            img_gray, detector, board, camera_matrix, dist_coeffs
        )
        if H_inv is None:
            print(f"  [{i+1}/{len(test_metadata)}] {img_name}: Failed ChArUco pose estimation. Skip.")
            continue

        # Model Inference
        img_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs)
            mask_resized = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255

        pred_mask = cv2.resize(mask_resized, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Contours & MinAreaRect
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"  [{i+1}/{len(test_metadata)}] {img_name}: Failed to segment target. Skip.")
            continue

        pred_mask_cleaned = np.zeros_like(pred_mask)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:
            print(f"  [{i+1}/{len(test_metadata)}] {img_name}: Segmented target area too small. Skip.")
            continue
            
        cv2.drawContours(pred_mask_cleaned, [largest_contour], -1, 255, -1)
        
        mask_indices = np.where(pred_mask_cleaned > 0)
        pixel_pts = np.column_stack((mask_indices[1], mask_indices[0]))

        # Unproject points
        N = len(pixel_pts)
        pts_hom = np.column_stack((pixel_pts, np.ones(N)))
        pts_board_hom = pts_hom @ H_inv.T
        pts_board = pts_board_hom[:, :2] / pts_board_hom[:, 2:3]

        # Calculate dimensions in mm
        rect = cv2.minAreaRect(pts_board.astype(np.float32))
        (cx_m, cy_m), (dim1_m, dim2_m), angle = rect
        measured_w = max(dim1_m, dim2_m) * 1000.0
        measured_h = min(dim1_m, dim2_m) * 1000.0

        # Calculate errors
        err_w = measured_w - GT_WIDTH_MM
        err_h = measured_h - GT_HEIGHT_MM
        pct_err_w = (abs(err_w) / GT_WIDTH_MM) * 100.0
        pct_err_h = (abs(err_h) / GT_HEIGHT_MM) * 100.0

        results.append({
            "image": img_name,
            "card_type": meta["card_type"],
            "pred_w": measured_w,
            "pred_h": measured_h,
            "err_w": err_w,
            "err_h": err_h,
            "pct_err_w": pct_err_w,
            "pct_err_h": pct_err_h
        })

        # Save an annotated image for visual check
        box_pts_board = cv2.boxPoints(rect)
        box_pts_board_3d = np.column_stack((box_pts_board, np.zeros(4)))
        img_box_pts, _ = cv2.projectPoints(
            box_pts_board_3d.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
        )
        img_box_pts = img_box_pts.reshape(-1, 2).astype(np.int32)

        annotated = img_undistorted.copy()
        cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners)
        overlay = annotated.copy()
        overlay[pred_mask_cleaned > 0] = [0, 255, 0]
        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
        cv2.drawContours(annotated, [img_box_pts], -1, (0, 0, 255), 8)
        box_center = np.mean(img_box_pts, axis=0).astype(np.int32)
        cv2.putText(annotated, f"W: {measured_w:.1f} mm", (box_center[0] - 150, box_center[1] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.putText(annotated, f"H: {measured_h:.1f} mm", (box_center[0] - 150, box_center[1] + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)
        cv2.imwrite(os.path.join(output_dir, img_name), annotated)

        print(f"  [{i+1}/{len(test_metadata)}] {img_name}: W={measured_w:.2f}mm, H={measured_h:.2f}mm")

    # 10. Generate Markdown Report & Table
    mae_w = np.mean([abs(r["err_w"]) for r in results])
    mae_h = np.mean([abs(r["err_h"]) for r in results])
    mpe_w = np.mean([r["pct_err_w"] for r in results])
    mpe_h = np.mean([r["pct_err_h"] for r in results])

    report_lines = []
    report_lines.append("# Accuracy Validation Report")
    report_lines.append("\nThis report evaluates the accuracy of the end-to-end measurement pipeline against physical ground-truth dimensions of the target Playing Card (**88.90 mm x 63.50 mm**).\n")
    report_lines.append("| Image File | Card Type | Measured Width | Measured Height | Width Error | Height Error | Width % Error | Height % Error |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        report_lines.append(f"| {r['image']} | {r['card_type']} | {r['pred_w']:.2f} mm | {r['pred_h']:.2f} mm | {r['err_w']:+.2f} mm | {r['err_h']:+.2f} mm | {r['pct_err_w']:.2f}% | {r['pct_err_h']:.2f}% |")
    
    report_lines.append("\n## Global Accuracy Summary")
    report_lines.append(f"- **Mean Absolute Error (MAE) - Width:** {mae_w:.4f} mm")
    report_lines.append(f"- **Mean Absolute Error (MAE) - Height:** {mae_h:.4f} mm")
    report_lines.append(f"- **Mean Percentage Error (MPE) - Width:** {mpe_w:.4f}%")
    report_lines.append(f"- **Mean Percentage Error (MPE) - Height:** {mpe_h:.4f}%")
    report_lines.append(f"- **Average Overall Error:** {((mpe_w + mpe_h) / 2.0):.4f}%")

    report_text = "\n".join(report_lines)
    
    # Save validation results JSON
    val_json_path = os.path.join(output_dir, "validation_results.json")
    with open(val_json_path, "w") as f:
        json.dump({
            "results": results,
            "metrics": {
                "mae_width": mae_w,
                "mae_height": mae_h,
                "mpe_width": mpe_w,
                "mpe_height": mpe_h
            }
        }, f, indent=4)

    # Save validation markdown report
    val_md_path = "measurements/accuracy_report.md"
    with open(val_md_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 50)
    print("Validation Complete!")
    print(f"  Mean Absolute Error (Width):  {mae_w:.3f} mm")
    print(f"  Mean Absolute Error (Height): {mae_h:.3f} mm")
    print(f"  Mean Percentage Error (Width):  {mpe_w:.3f}%")
    print(f"  Mean Percentage Error (Height): {mpe_h:.3f}%")
    print(f"Validation Report saved to: {val_md_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
