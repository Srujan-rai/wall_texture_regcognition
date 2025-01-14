import cv2
import numpy as np
import os

def enhance_texture(image_path, output_path):
    
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)
    
    smoothed_image = cv2.bilateralFilter(equalized_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)
    
    edges = cv2.Canny(sharpened_image, 50, 150)
    
    combined_image = cv2.addWeighted(sharpened_image, 0.8, edges, 0.2, 0)
    
    cv2.imwrite(output_path, combined_image)

def process_dataset(input_dir, output_dir):
  
    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, file)
                
                print(f"Processing {input_path} -> {output_path}")
                enhance_texture(input_path, output_path)

input_dataset_dir = "fyp robot.v3i.yolov8" 
output_dataset_dir = "fyp_robot_transformed" 

process_dataset(input_dataset_dir, output_dataset_dir)
