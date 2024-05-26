import cv2
import numpy as np
import os

def calculate_roundness(contour):
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # Calculate the roundness factor
    if perimeter != 0:
        roundness_factor = (4 * np.pi * area) / (perimeter ** 2)
        return roundness_factor
    else:
        return None 

def convert_to_cm2(area_px, pixels_per_cm):
    # Convert area from pixels to square centimeters
    area_cm2 = area_px / (pixels_per_cm ** 2)
    return area_cm2

def detect_and_measure_contours(image, min_area_threshold, pixels_per_cm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(blurred)

    # Thresholding for white regions
    _, thresh_white = cv2.threshold(equalized, 240, 255, cv2.THRESH_BINARY)

    # Thresholding for gray regions
    _, thresh_gray = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove small noises for white regions
    kernel = np.ones((3, 3), np.uint8)
    morphed_white = cv2.morphologyEx(thresh_white, cv2.MORPH_CLOSE, kernel)

    # Apply morphological operations to remove small noises for gray regions
    morphed_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary images
    contours_white, _ = cv2.findContours(morphed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_gray, _ = cv2.findContours(morphed_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate over white contours
    for contour in contours_white:
        # Calculate roundness factor
        roundness = calculate_roundness(contour)
        
        # Convert area to square centimeters
        area_px = cv2.contourArea(contour)
        area_cm2 = convert_to_cm2(area_px, pixels_per_cm)
        
        # Filter out small contours
        if roundness is not None and area_cm2 is not None and area_px >= min_area_threshold:
            # Draw contour on the original image
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate dimensions in centimeters
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            
            # Print the values to the terminal
            print(f'White Region - Roundness: {roundness:.2f}, Area: {area_cm2:.2f} cm^2, Width: {width_cm:.2f} cm, Height: {height_cm:.2f} cm')

    # Iterate over gray contours
    for contour in contours_gray:
        # Calculate roundness factor
        roundness = calculate_roundness(contour)
        
        # Convert area to square centimeters
        area_px = cv2.contourArea(contour)
        area_cm2 = convert_to_cm2(area_px, pixels_per_cm)
        
        # Filter out small contours
        if roundness is not None and area_cm2 is not None and area_px >= min_area_threshold:
            # Draw contour on the original image
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate dimensions in centimeters
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            
            # Print the values to the terminal
            print(f'Gray Region - Roundness: {roundness:.2f}, Area: {area_cm2:.2f} cm^2, Width: {width_cm:.2f} cm, Height: {height_cm:.2f} cm')

    return image

# Load an image
img = cv2.imread(r"C:\Users\rauna\Desktop\fragments.jpg")

# Set pixels per centimeter based on given information
pixels_per_cm = 10.92 * 10  # 10.92 pixels per millimeter, 109.2 pixels per centimeter

# Minimum contour area threshold
min_area_threshold = 1000  # Adjust as needed

# Detect and measure contours
result_img = detect_and_measure_contours(img, min_area_threshold, pixels_per_cm)

# Save the contoured image on the desktop
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
output_path = os.path.join(desktop_path, 'contoured_image.jpg')
cv2.imwrite(output_path, result_img)

# Display the image with contours
cv2.imshow('Contours and Measurements', result_img)

# Ask the user if they are satisfied with the output image
satisfaction = input("Are you satisfied with the output image? (yes/no): ")
if satisfaction.lower() == "yes":
    print("Great! Image saved on desktop.")
else:
    print("Please adjust parameters to improve results.")

cv2.waitKey(0)
cv2.destroyAllWindows()



















