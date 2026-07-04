import os
import json
import glob
import cv2
import numpy as np
import random

def create_card_template(card_type="spade"):
    h, w = 1120, 800
    card = np.ones((h, w, 3), dtype=np.uint8) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw rounded mask
    r = 40  # corner radius
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    
    # Draw card body: off-white
    card[mask == 255] = [245, 245, 245]
    
    # Draw a thin dark border inside
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_r = 35
    cv2.rectangle(border_mask, (border_r, 5), (w - border_r, h - 5), 255, -1)
    cv2.rectangle(border_mask, (5, border_r), (w - 5, h - border_r), 255, -1)
    cv2.circle(border_mask, (border_r, border_r), border_r, 255, -1)
    cv2.circle(border_mask, (w - border_r, border_r), border_r, 255, -1)
    cv2.circle(border_mask, (border_r, h - border_r), border_r, 255, -1)
    cv2.circle(border_mask, (w - border_r, h - border_r), border_r, 255, -1)
    
    card[mask != border_mask] = [50, 50, 50]  # dark grey border
    
    if card_type == "spade":
        color = (0, 0, 0)  # black
        # Draw central spade
        cv2.circle(card, (350, 580), 80, color, -1)
        cv2.circle(card, (450, 580), 80, color, -1)
        pts_tri = np.array([[400, 360], [300, 530], [500, 530]], dtype=np.int32)
        cv2.fillPoly(card, [pts_tri], color)
        pts_stem = np.array([[400, 560], [360, 750], [440, 750]], dtype=np.int32)
        cv2.fillPoly(card, [pts_stem], color)
        
        # Add corners "A"
        cv2.putText(card, "A", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 8)
        # Small spade
        cv2.circle(card, (55, 170), 12, color, -1)
        cv2.circle(card, (70, 170), 12, color, -1)
        pts_sm_tri = np.array([[62, 135], [47, 160], [77, 160]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_tri], color)
        pts_sm_stem = np.array([[62, 165], [57, 195], [67, 195]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_stem], color)
        
        # Rotated corner (bottom-right)
        cv2.putText(card, "A", (w - 110, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 8)
        cv2.circle(card, (w - 70, h - 170), 12, color, -1)
        cv2.circle(card, (w - 55, h - 170), 12, color, -1)
        pts_sm_tri2 = np.array([[w - 62, h - 135], [w - 47, h - 160], [w - 77, h - 160]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_tri2], color)
        pts_sm_stem2 = np.array([[w - 62, h - 165], [w - 57, h - 195], [w - 67, h - 195]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_stem2], color)
        
    elif card_type == "heart":
        color = (0, 0, 200)  # red
        # Draw central heart
        cv2.circle(card, (350, 500), 80, color, -1)
        cv2.circle(card, (450, 500), 80, color, -1)
        pts_tri = np.array([[400, 720], [285, 545], [515, 545]], dtype=np.int32)
        cv2.fillPoly(card, [pts_tri], color)
        
        # Corners "K"
        cv2.putText(card, "K", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 8)
        cv2.putText(card, "K", (w - 110, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 8)
        # Small heart
        cv2.circle(card, (55, 160), 12, color, -1)
        cv2.circle(card, (70, 160), 12, color, -1)
        pts_sm_tri = np.array([[62, 195], [47, 170], [77, 170]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_tri], color)
        
        cv2.circle(card, (w - 70, h - 160), 12, color, -1)
        cv2.circle(card, (w - 55, h - 160), 12, color, -1)
        pts_sm_tri2 = np.array([[w - 62, h - 195], [w - 47, h - 170], [w - 77, h - 170]], dtype=np.int32)
        cv2.fillPoly(card, [pts_sm_tri2], color)
        
    return card, mask

def main():
    random.seed(42)
    np.random.seed(42)
    
    # 1. Load Calibration Parameters
    with open(os.path.join("calibration", "camera_params.json"), "r") as f:
        camera_params = json.load(f)
    
    camera_matrix = np.array(camera_params["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(camera_params["distortion_coefficients"], dtype=np.float64)
    
    # 2. Setup ChArUco Board Setup for Pose Estimation
    squares_x = 8
    squares_y = 10
    square_length = 0.024  # from generate_board.py
    marker_length = 0.018
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_length, 
        marker_length, 
        aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(board)
    
    # 3. Load Background/Calibration Images
    image_paths = sorted(glob.glob(os.path.join("calibration", "images", "*.jpg")))
    # Deduplicate paths
    image_paths = sorted(list(set(os.path.abspath(p) for p in image_paths)))
    
    print(f"Loaded {len(image_paths)} calibration backgrounds.")
    
    # 4. Generate Playing Card Templates
    card_spade, mask_spade = create_card_template("spade")
    card_heart, mask_heart = create_card_template("heart")
    
    card_w, card_h = 800, 1120
    # Source corners of the card template
    src_pts = np.array([
        [0, 0],
        [card_w - 1, 0],
        [card_w - 1, card_h - 1],
        [0, card_h - 1]
    ], dtype=np.float32)
    
    # Playing Card Physical Dimensions in meters
    card_width_m = 0.0889
    card_height_m = 0.0635
    
    # We will output train, val, test splits
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Determine split index mapping (approx 70% train / 20% val / 10% test)
    # Total targets: we want around 100 images. Let's do 4 variations for each of the 29 images = 116 images.
    dataset_samples = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco board
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 6:
            continue
            
        # Solve Pose (PnP) using detected chessboard corners and their 3D coordinates
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
            continue
            
        # Generate 4 random placements of card per image
        for idx in range(4):
            # Chessboard is 8x10 squares. Total size = (8*0.024) x (10*0.024) = 0.192 x 0.240 meters.
            # Card size is 0.0889 x 0.0635.
            # Place card somewhere on the board, leaving some margins
            # We want to randomize position and rotation in the plane (z=0)
            cx = random.uniform(0.04, 0.10)
            cy = random.uniform(0.04, 0.13)
            theta = random.uniform(-np.pi/4, np.pi/4)  # rotation angle in radians
            
            # Compute 3D corners of the card flat on the board (z=0)
            c_w2 = card_width_m / 2.0
            c_h2 = card_height_m / 2.0
            local_corners = np.array([
                [-c_w2, -c_h2, 0.0],
                [c_w2, -c_h2, 0.0],
                [c_w2, c_h2, 0.0],
                [-c_w2, c_h2, 0.0]
            ], dtype=np.float32)
            
            R_z = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            card_pts_3d = []
            for pt in local_corners:
                rotated_pt = R_z.dot(pt)
                translated_pt = rotated_pt + np.array([cx, cy, 0.0])
                card_pts_3d.append(translated_pt)
            card_pts_3d = np.array(card_pts_3d, dtype=np.float32)
            
            # Project 3D points back to 2D image plane
            pts_img, _ = cv2.projectPoints(card_pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
            pts_img = pts_img.reshape(-1, 2)
            
            # Check if all projected corners are within image bounds
            h_img, w_img = img.shape[:2]
            if np.any(pts_img[:, 0] < 0) or np.any(pts_img[:, 0] >= w_img) or \
               np.any(pts_img[:, 1] < 0) or np.any(pts_img[:, 1] >= h_img):
                continue
                
            # Randomly pick card type
            card_type = random.choice(["spade", "heart"])
            card_template, mask_template = (card_spade, mask_spade) if card_type == "spade" else (card_heart, mask_heart)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(src_pts, pts_img.astype(np.float32))
            
            # Warp card and mask
            warped_card = cv2.warpPerspective(card_template, M, (w_img, h_img))
            warped_mask = cv2.warpPerspective(mask_template, M, (w_img, h_img), flags=cv2.INTER_NEAREST)
            
            # Soften the mask edge by blending
            kernel_size = 5
            alpha = warped_mask.astype(np.float32) / 255.0
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
            alpha = np.expand_dims(alpha, axis=2)
            
            # Blend card into the image
            blended = img.astype(np.float32) * (1.0 - alpha) + warped_card.astype(np.float32) * alpha
            blended = blended.astype(np.uint8)
            
            brightness_factor = random.uniform(0.85, 1.15)
            card_mask_bool = warped_mask > 0
            if brightness_factor != 1.0:
                blended_adjust = blended.astype(np.float32)
                blended_adjust[card_mask_bool] *= brightness_factor
                blended = np.clip(blended_adjust, 0, 255).astype(np.uint8)
            
            dataset_samples.append({
                "image": blended,
                "mask": warped_mask,
                "card_type": card_type,
                "cx": cx,
                "cy": cy,
                "theta": theta,
                "corners_2d": pts_img.tolist(),
                "bg_image": os.path.basename(path)
            })

    print(f"Generated {len(dataset_samples)} valid synthetic image samples.")
    
    random.shuffle(dataset_samples)
    total_samples = len(dataset_samples)
    
    train_idx = int(0.70 * total_samples)
    val_idx = int(0.90 * total_samples)
    
    train_samples = dataset_samples[:train_idx]
    val_samples = dataset_samples[train_idx:val_idx]
    test_samples = dataset_samples[val_idx:]
    
    split_data = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    
    for split_name, samples in split_data.items():
        img_dir = os.path.join("dataset", split_name, "images")
        mask_dir = os.path.join("dataset", split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        metadata = []
        for i, sample in enumerate(samples):
            img_filename = f"sample_{i:04d}.png"
            mask_filename = f"sample_{i:04d}_mask.png"
            
            cv2.imwrite(os.path.join(img_dir, img_filename), sample["image"])
            cv2.imwrite(os.path.join(mask_dir, mask_filename), sample["mask"])
            
            metadata.append({
                "image_file": img_filename,
                "mask_file": mask_filename,
                "card_type": sample["card_type"],
                "chess_coord_cx_m": sample["cx"],
                "chess_coord_cy_m": sample["cy"],
                "chess_coord_theta_rad": sample["theta"],
                "corners_2d": sample["corners_2d"],
                "bg_image": sample["bg_image"]
            })
            
        with open(os.path.join("dataset", split_name, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Saved {len(samples)} samples to dataset/{split_name}/")

if __name__ == "__main__":
    main()
