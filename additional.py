import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def enhance_texture_single(image_path, output_path):
    """
    Enhance the texture of a wall surface in a single image while managing light intensity variations.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)
    
    # Apply bilateral filter to reduce noise and preserve edges
    smoothed_image = cv2.bilateralFilter(equalized_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)
    
    # Perform edge detection
    edges = cv2.Canny(sharpened_image, 50, 150)
    
    # Combine sharpened image with edges
    combined_image = cv2.addWeighted(sharpened_image, 0.8, edges, 0.2, 0)
    
    # Save the result
    cv2.imwrite(output_path, combined_image)
    
    # Display the original and enhanced images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Texture Image")
    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

    return combined_image  # Return the enhanced image for further processing


def compute_metrics(image):
    """
    Compute multiple texture metrics for an image.
    """
    # Compute edges
    edges = cv2.Canny(image, 50, 150)
    edge_count = np.sum(edges > 0)
    
    # Compute GLCM (Gray-Level Co-occurrence Matrix) contrast
    glcm = greycomatrix(image, [1], [0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    
    # Compute entropy
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize the histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
    
    # Compute mean and standard deviation
    mean = np.mean(image)
    std_dev = np.std(image)
    
    return {
        "edge_count": edge_count,
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity,
        "entropy": entropy,
        "mean": mean,
        "std_dev": std_dev
    }


def classify_quality(metrics, thresholds):
    """
    Classify the wall finish quality based on computed metrics and thresholds.
    """
    # Implement your logic to classify based on thresholds for multiple metrics
    if (metrics["edge_count"] < thresholds["edge_count"] and
        metrics["contrast"] < thresholds["contrast"] and
        metrics["correlation"] > thresholds["correlation"] and
        metrics["energy"] > thresholds["energy"] and
        metrics["homogeneity"] > thresholds["homogeneity"] and
        metrics["entropy"] < thresholds["entropy"]):
        return "Good"
    
    return "Bad"


# Paths for input and output images
input_image_path = "good_bad_conversion/dataset/train/images/bad/20240122_101958_jpg.rf.830610f1666bf8590f1881028b33c8ff.jpg"  
output_image_path = "enhanced_texture_image.jpg"  

# Process the single image
enhanced_image = enhance_texture_single(input_image_path, output_image_path)

metrics = compute_metrics(enhanced_image)
print(f"Computed Metrics: {metrics}")

thresholds = {
    "edge_count": 750, 
    "contrast": 50, 
    "correlation": 0.5,  
    "energy": 0.05,     
    "homogeneity": 0.2,  
    "entropy": 6       
}  

# Classify the quality
quality = classify_quality(metrics, thresholds)
print(f"Wall Finish Quality: {quality}")
