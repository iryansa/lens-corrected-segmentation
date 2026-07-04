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

def main():
    parser = argparse.ArgumentParser(description="Predict segmentation mask for an image using the trained U-Net model.")
    parser.add_argument("--image", type=str, required=True, help="Path to input raw image.")
    parser.add_argument("--output", type=str, default="inference/output.png", help="Path to save the annotated output.")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Path to model weights.")
    parser.add_argument("--camera-params", type=str, default="calibration/camera_params.json", help="Path to camera_params.json.")
    args = parser.parse_args()

    # 1. Undistort the input image using camera calibration parameters
    print(f"Loading camera parameters from {args.camera_params}...")
    camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
    
    print(f"Loading raw image: {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at {args.image}")
        return
        
    print("Applying camera undistortion...")
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # 2. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # initialization doesn't need pre-trained weights for loading
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Preprocess the image for the model
    h_orig, w_orig = img_undistorted.shape[:2]
    # Resize to 512x512
    img_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 4. Model Inference
    print("Running model inference...")
    with torch.no_grad():
        outputs = model(img_tensor)
        # Apply sigmoid to logits
        probs = torch.sigmoid(outputs)
        pred_mask_resized = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255

    # 5. Postprocess mask back to original resolution
    pred_mask = cv2.resize(pred_mask_resized, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # 6. Generate annotated output overlay
    overlay = img_undistorted.copy()
    # Draw green mask overlay (translucent)
    overlay[pred_mask > 0] = [0, 255, 0]
    alpha = 0.4
    annotated = cv2.addWeighted(overlay, alpha, img_undistorted, 1.0 - alpha, 0)

    # Add text banner
    cv2.putText(annotated, "Segmented Object Mask", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

    # Save to disk
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, annotated)
    print(f"Success! Annotated prediction saved to: {args.output}")

if __name__ == "__main__":
    main()
