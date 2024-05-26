import cv2
import numpy as np

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

# Load the image
image = cv2.imread(r"C:\Users\rauna\Desktop\contoured_image.jpg")

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

# Define range for blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only green and blue colors
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours for green color
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find contours for blue color
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours_green, -1, (0, 255, 0), 2)
cv2.drawContours(image, contours_blue, -1, (255, 0, 0), 2)

# Define pixels per centimeter conversion factor
pixels_per_cm = 8  # 1mm as 8 pixels

# Calculate roundness and perimeter in cm² and cm
for contour in contours_green:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    roundness = calculate_roundness(contour)
    if roundness is not None:
        area_cm2 = convert_to_cm2(area, pixels_per_cm)
        perimeter_cm = perimeter / pixels_per_cm
        print("Green Contour - Perimeter:", perimeter_cm, "cm, Roundness:", roundness, "Area:", area_cm2, "cm²")

for contour in contours_blue:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    roundness = calculate_roundness(contour)
    if roundness is not None:
        area_cm2 = convert_to_cm2(area, pixels_per_cm)
        perimeter_cm = perimeter / pixels_per_cm
        print("Blue Contour - Perimeter:", perimeter_cm, "cm, Roundness:", roundness, "Area:", area_cm2, "cm²")

# Show the image
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




