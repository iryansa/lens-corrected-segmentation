import os
import json
import glob
import cv2
import numpy as np

def main():
    # 1. Match the exact structural parameters used to generate your board
    squares_x = 8
    squares_y = 10
    square_length = 0.02005  # 20.05 mm converted to meters
    marker_length = 0.01503  # 15.03 mm converted to meters

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_length, 
        marker_length, 
        aruco_dict
    )

    # Initialize the modern OpenCV ChArUco Detector API
    detector = cv2.aruco.CharucoDetector(board)

    # 2. Prepare collection arrays
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    # Find all images inside your calibration folder (handles png, jpg, jpeg)
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join("calibration", "images", ext)))
    
    print(f"Found {len(image_paths)} calibration images to process.")

    if len(image_paths) < 20:
        print("Warning: Guidelines recommend a minimum of 20 images for accuracy!")

    # 3. Process every captured frame
    for path in sorted(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            # Capture base pixel width and height of your camera sensor
            image_size = (gray.shape[1], gray.shape[0])

        # Automatically locate the ChArUco board layout
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        # Check if enough keypoints were extracted cleanly in this frame
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"Extracted data successfully from: {os.path.basename(path)}")
        else:
            print(f"Failed to parse full board from: {os.path.basename(path)} (Skip frame)")

    # 4. Perform Camera Calibration
    if len(all_charuco_corners) >= 3:
        print("\nRunning intrinsic geometric matrix operations...")
        
        # This function generates your internal spatial profile
        ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        # 5. Review performance metric and output configuration parameters
        print("-" * 50)
        print(f"Calibration Complete!")
        print(f"Reprojection Error (MAE Pixels): {ret:.4f}")
        
        # A value under 0.3 pixels indicates outstanding tracking accuracy
        if ret < 0.3:
            print("Evaluation Status: Excellent precision standard!")
        elif ret < 0.5:
            print("Evaluation Status: Acceptable operational baseline.")
        else:
            print("Evaluation Status: Error too high. Check if paper is flat or blurry.")
        print("-" * 50)

        # Format numerical matrices to structured Python lists for storage
        calibration_data = {
            "reprojection_error": float(ret),
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_size": image_size
        }

        # Save to disk as reusable parameters
        output_path = os.path.join("calibration", "camera_params.json")
        with open(output_path, "w") as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"Configuration profile written to: {output_path}")

    else:
        print("\nError: Could not extract clear board data across enough frames.")
        print("Ensure camera shots are clear, sharp, well-lit, and capture the grid cleanly.")

if __name__ == "__main__":
    main()