from tkinter import Canvas, simpledialog
import cv2
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Define all necessary functions
def calculate_roundness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter != 0:
        roundness_factor = (4 * np.pi * area) / (perimeter ** 2)
        return roundness_factor
    else:
        return None

def convert_to_cm2(area_px, pixels_per_cm):
    area_cm2 = area_px / (pixels_per_cm ** 2)
    return area_cm2

def convert_to_mm2(area_px, pixels_per_mm):
    area_mm2 = area_px / (pixels_per_mm ** 2)
    return area_mm2

def calculate_average_diameter(contour):
    _, radius = cv2.minEnclosingCircle(contour)
    average_diameter = 2 * radius
    return average_diameter

def calculate_volume(contour, roundness, width_mm, height_mm):
    if roundness >= 0.7:
        radius_mm = width_mm / 2
        volume_mm3 = (4/3) * np.pi * (radius_mm ** 3)
    else:
        a = width_mm
        b = height_mm / 2
        volume_mm3 = (4/3) * np.pi * a * (b ** 2)
    return volume_mm3

def calculate_mass(volume_mm3, density_g_per_cm3):
    density_g_per_mm3 = density_g_per_cm3 / 1000
    mass_g = volume_mm3 * density_g_per_mm3
    return mass_g

def calculate_height_width(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w, h

def detect_and_measure_contours(image, min_area_threshold, pixels_per_cm):
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    _, thresh_white = cv2.threshold(equalized, 240, 255, cv2.THRESH_BINARY)
    _, thresh_gray = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    morphed_white = cv2.morphologyEx(thresh_white, cv2.MORPH_CLOSE, kernel)
    morphed_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
    contours_white, _ = cv2.findContours(morphed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_gray, _ = cv2.findContours(morphed_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours_white + contours_gray:
        roundness = calculate_roundness(contour)
        area_px = cv2.contourArea(contour)
        area_cm2 = convert_to_cm2(area_px, pixels_per_cm)
        if roundness is not None and area_cm2 is not None and area_px >= min_area_threshold:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            width_cm = w / pixels_per_cm
            height_cm = h / pixels_per_cm
            results.append(f'Roundness: {roundness:.2f}, Area: {area_cm2:.2f} cm^2, Width: {width_cm:.2f} cm, Height: {height_cm:.2f} cm')
    return image, results

def display_results(results):
    root = tk.Tk()
    root.title("Contour Analysis Results")
    text_widget = tk.Text(root, wrap='word', width=100, height=30)
    text_widget.pack(expand=True, fill='both')
    for result in results:
        text_widget.insert('end', result + '\n')
    def check_satisfaction():
        response = messagebox.askyesno("Satisfaction Check", "Are you satisfied with the output image?")
        if response:
            messagebox.showinfo("Feedback", "Great! Image saved on desktop.")
        else:
            messagebox.showinfo("Feedback", "The image has been saved. Please adjust parameters to improve results.")
        root.destroy()
        if not response:
            analyze_image()  # Jump to Extract Data from Image if not satisfied
    button = tk.Button(root, text="Check Satisfaction", command=check_satisfaction)
    button.pack(pady=10)
    root.mainloop()

def crop_image(img):
    r = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    if r != (0, 0, 0, 0):
        img_cropped = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.destroyWindow("Select ROI")
        return img_cropped
    else:
        cv2.destroyWindow("Select ROI")
        return img
    
def perform_homography(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) == 4:
        src_points = np.array([point[0] for point in approx], dtype=np.float32)
    else:
        raise Exception("The largest contour is not a quadrilateral.")
    width, height = 1000, 700
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    img_warped = cv2.warpPerspective(img, H, (width, height))

    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    output_path = os.path.join(desktop_path, 'calibrated_image.jpg')
    if os.path.isdir(desktop_path) and cv2.imwrite(output_path, img_warped):
        print(f"Calibrated image successfully saved to: {output_path}")
    else:
        print("Failed to save the calibrated image.")

    return img_warped

def analyze_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png; *.jpg; *.jpeg")])
    if file_path:
        img = cv2.imread(file_path)
        img = crop_image(img)
        pixels_per_cm = 8 * 10
        min_area_threshold = 10
        result_img, results = detect_and_measure_contours(img, min_area_threshold, pixels_per_cm)
        cv2.imshow('Contours and Measurements', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
        output_path = os.path.join(desktop_path, 'contoured_image.jpg')
        if os.path.isdir(desktop_path) and cv2.imwrite(output_path, result_img):
            print(f"Image successfully saved to: {output_path}")
        else:
            print("Failed to save the image.")
        display_results(results)

def run_calibration():
    image_path = filedialog.askopenfilename(title="Select Image for Calibration", filetypes=[("Image Files", "*.png; *.jpg; *.jpeg")])
    if not image_path:
        print("No image selected. Exiting the function.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image. Please check the path and try again.")
        return
    calibrated_img = perform_homography(img)
    if calibrated_img is not None:
        cv2.imshow('Calibrated Image', calibrated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def additional_analysis():
    image_path = filedialog.askopenfilename(title="Select Contoured Image", filetypes=[("Image Files", "*.png; *.jpg; *.jpeg")])
    if not image_path:
        print("No image selected. Exiting the function.")
        return
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Please check the path and try again.")
        return
    


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours_green, -1, (0, 255, 0), thickness=cv2.FILLED)
    cv2.drawContours(image, contours_red, -1, (0, 0, 255), thickness=cv2.FILLED)
    pixels_per_mm = 8
    density_g_per_cm3 = float(simpledialog.askstring("Input", "Please enter the density of the material (in g/cm^3):"))

    contour_details = []

    for color, contours in zip(["Green", "Red"], [contours_green, contours_red]):
        for contour_index, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            roundness = calculate_roundness(contour)
            if roundness is not None:
                area_mm2 = convert_to_mm2(area, pixels_per_mm)
                perimeter_mm = perimeter / pixels_per_mm
                average_diameter = calculate_average_diameter(contour) / pixels_per_mm
                width_px, height_px = calculate_height_width(contour)
                width_mm = width_px / pixels_per_mm
                height_mm = height_px / pixels_per_mm
                volume_mm3 = calculate_volume(contour, roundness, width_mm, height_mm)
                mass_g = calculate_mass(volume_mm3, density_g_per_cm3)
                contour_details.append({
                    "Color": color,
                    "Contour Index": contour_index + 1,
                    "Perimeter (mm)": perimeter_mm,
                    "Area (mm^2)": area_mm2,
                    "Roundness": roundness,
                    "Average Diameter (mm)": average_diameter,
                    "Width (mm)": width_mm,
                    "Height (mm)": height_mm,
                    "Volume (mm^3)": volume_mm3,
                    "Mass (g)": mass_g
                })

    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    output_file = os.path.join(desktop_path, "contour_details.xlsx")
    df = pd.DataFrame(contour_details)
    df.to_excel(output_file, index=False)
    if os.path.isfile(output_file):
        print(f"\nContour details successfully saved to: {output_file}")
        csv_output_file = os.path.splitext(output_file)[0] + '.csv'
        try:
            df.to_csv(csv_output_file, index=False)
            print(f"Contour details successfully converted and saved to CSV: {csv_output_file}")
        except Exception as e:
            print(f"Failed to convert and save contour details to CSV: {str(e)}")

      

    else:
        print("\nFailed to save contour details.")

def perform_sieve_analysis():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        print("No file selected. Exiting the function.")
        return

    df = pd.read_csv(file_path)

    # Check if required columns exist
    if 'Average Diameter (mm)' not in df.columns or 'Mass (g)' not in df.columns:
        raise ValueError("The required columns are not present in the CSV file")

    # Find the range of average diameter and divide it into a scale of 20
    min_diameter = df['Average Diameter (mm)'].min()
    max_diameter = df['Average Diameter (mm)'].max()
    bin_edges = np.logspace(np.log10(min_diameter), np.log10(max_diameter), num=101)  # 20 intervals

    # Categorize average diameters into sieve sizes
    df['Sieve Size (mm)'] = pd.cut(df['Average Diameter (mm)'], bins=bin_edges, labels=bin_edges[:-1], include_lowest=True)

    # Calculate the total mass
    total_mass = df['Mass (g)'].sum()

    # Aggregate data for sieve summary
    sieve_summary = df.groupby('Sieve Size (mm)').agg({'Mass (g)': 'sum'}).sort_index()
    sieve_summary['Cumulative Mass (g)'] = sieve_summary['Mass (g)'].cumsum()
    sieve_summary['% Passing'] = sieve_summary['Cumulative Mass (g)'] / total_mass * 100
    sieve_summary['% Retained'] = 100 - sieve_summary['% Passing']

    # Function to find D10, D30, and D60
    def find_D_value(percent_passing, target_percent):
        return np.interp(target_percent, percent_passing, bin_edges[:-1])

    # Find D10, D30, and D60
    D10 = find_D_value(sieve_summary['% Passing'], 10)
    D30 = find_D_value(sieve_summary['% Passing'], 30)
    D60 = find_D_value(sieve_summary['% Passing'], 60)

    # Calculate the coefficient of uniformity (Cu)
    Cu = D60 / D10

    # Calculate the coefficient of curvature (Cc)
    Cc = (D30 ** 2) / (D10 * D60)

    # Display the coefficient of uniformity and coefficient of curvature with interpretation
    print(f"Coefficient of Uniformity (Cu): {Cu:.2f}")
    print(f"Coefficient of Curvature (Cc): {Cc:.2f}")

    # Interpret the Coefficient of Uniformity (Cu)
    if Cu == 1:
        print("Cu = 1 indicates a soil with only one grain size.")
    elif 2 <= Cu <= 3:
        print("Cu between 2 and 3 indicates very poorly graded soils, such as beach sands.")
    elif Cu >= 15:
        print("Cu of 15 or greater indicates very well graded soils.")
    elif 400 <= Cu <= 500:
        print("Cu between 400 and 500 indicates sizes ranging from large boulders to very fine-grained clay particles.")
    else:
        print("Cu indicates a soil with moderate grading.")

    # Interpret the Coefficient of Curvature (Cc)
    if 1 <= Cc <= 3:
        print("Cc between 1 and 3 indicates a well-graded soil.")
    else:
        print("Cc indicates a poorly graded soil.")

    # Display the DataFrame with the new columns
    print(sieve_summary)

    # Save the updated DataFrame to a new CSV file
    output_csv_path = os.path.join(os.path.expanduser("~"), 'Desktop', 'sieve_summary.csv')
    sieve_summary.to_csv(output_csv_path, index=True)

    # Plotting the gradation curve and histogram
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting Passing Percentage Curve
    ax1.plot(bin_edges[:-1], sieve_summary['% Passing'], marker='o', linestyle='-', color='b', label='% Passing')
    ax1.set_xlabel('Particle Diameter (mm)')
    ax1.set_ylabel('Cumulative Percentage Passing (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.invert_xaxis()  # Ensure the x-axis shows from maximum to 0
    ax1.set_xscale('log')  # Set x-axis to logarithmic scale
    ax1.set_xlim(left=bin_edges.min(), right=bin_edges.max())

    # Plotting Retained Percentage Curve
    ax1.plot(bin_edges[:-1], sieve_summary['% Retained'], marker='o', linestyle='-', color='r', label='% Retained')
    ax1.legend(loc='upper left')

    # Adding annotations for Cu and Cc
    textstr = '\n'.join((
        f'Cu: {Cu:.2f}',
        f'Cc: {Cc:.2f}',
        '1 <= Cc <= 3: Well-graded soil'))

    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place the text box in the plot
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    # Plotting histogram
    ax2 = ax1.twinx()
    ax2.bar(sieve_summary.index.astype(float), sieve_summary['Mass (g)'], width=0.1, alpha=0.6, color='green', label='Mass (g)')
    ax2.set_ylabel('Mass (g)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Adding a distribution curve to the histogram
    mass_distribution = np.repeat(sieve_summary.index.astype(float).values, sieve_summary['Mass (g)'].astype(int))
    density, bins, _ = ax2.hist(mass_distribution, bins=bin_edges, density=True, alpha=0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax2.plot(bin_centers, density, color='black', linewidth=2, label='Distribution')

    fig.tight_layout()
    plt.show()

    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    output_file = os.path.join(desktop_path, "sieve_analysis.xlsx")
    sieve_df.to_excel(output_file, index=False)
    if os.path.isfile(output_file):
        print(f"\nSieve analysis results successfully saved to: {output_file}")
    else:
        print("\nFailed to save sieve analysis results.")

def create_additional_buttons(root, canvas):
    button_calibrate = tk.Button(root, text="Calibration", command=run_calibration, width=25, height=2, padx=10, pady=2)
    canvas.create_window(500, 550, anchor="nw", window=button_calibrate)

# Define the GUI application
def main():
    root = tk.Tk()
    root.title("Image Analysis Tool")
    
    # Load and set the background image
    bg_image = Image.open(r"C:\Users\rauna\Desktop\themepic.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = tk.Canvas(root, width=bg_photo.width(), height=bg_photo.height())
    canvas.pack(fill="both", expand=True)
    bg= canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    def run_image_analysis():
        analyze_image()

    def create_additional_buttons(root, canvas):
        button_calibrate = tk.Button(root, text="Calibration", command=run_calibration, width=10, height=1, padx=10, pady=2)
        canvas.create_window(550, 350, anchor="nw", window=button_calibrate)


    def run_additional_analysis():
        additional_analysis()


    def run_sieve_analysis():
        perform_sieve_analysis()


    heading_label = tk.Label(root, text="FragmentGrade Dynamics", font=("Helvetica", 30, "bold"),bg="black", fg="white")
    canvas.create_window(400, 50, anchor="nw", window=heading_label)

    button1 = tk.Button(root, text="Image Contour Analysis", command=run_image_analysis,width=25, height=2,padx=15, pady=2)
    button2 = tk.Button(root, text="Extract Data from Image", command=run_additional_analysis,width=25, height=2,padx=15, pady=2)
    button3 = tk.Button(root, text="Gradation Curve", command=run_sieve_analysis,width=25, height=2,padx=15, pady=2)

    canvas.create_window(500, 400, anchor="nw", window=button1)
    canvas.create_window(500, 450, anchor="nw", window=button2)
    canvas.create_window(500, 500, anchor="nw", window=button3)

    root.geometry(f"{bg_photo.width()}x{bg_photo.height()}")

    create_additional_buttons(root,canvas)
    root.mainloop()

if __name__ == "__main__":
    main()

