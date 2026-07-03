import cv2

def main():
    # 1. Initialize the predefined ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    # 2. Define structural dimensions to satisfy the "7x9 inner corners" rule
    # 8 columns x 10 rows of squares yield exactly (8-1)x(10-1) = 7x9 inner corners
    squares_x = 8
    squares_y = 10

    # Physical dimensions scaled for A4 / US Letter portrait layout
    # 0.024m = 24mm square length | 0.018m = 18mm internal marker length
    square_length = 0.024 
    marker_length = 0.018

    # 3. Create the ChArUco Board Object (Modern OpenCV 4.7+ API)
    print("Initializing ChArUco Board with 7x9 inner corners...")
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_length, 
        marker_length, 
        aruco_dict
    )

    # 4. Generate a high-resolution image for printing
    # Adjusted to 2000x2500 to match the elongated 8:10 vertical aspect ratio
    print("Rendering high-res pattern...")
    board_image = board.generateImage((2000, 2500), marginSize=60)

    # 5. Export to disk
    output_filename = "charuco_board_7x9_corners.png"
    cv2.imwrite(output_filename, board_image)
    
    print("-" * 50)
    print(f"Success! Board generated and saved as: {output_filename}")
    print(f"Settings to log in your CALIBRATION_REPORT.md:")
    print(f" -> Columns (Squares X): {squares_x}, Rows (Squares Y): {squares_y}")
    print(f" -> Resulting Inner Corners: {squares_x - 1} x {squares_y - 1}")
    print(f" -> Physical Square Size: {square_length * 1000} mm")
    print(f" -> Physical Marker Size: {marker_length * 1000} mm")
    print("-" * 50)

if __name__ == "__main__":
    main()