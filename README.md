

# Wall Quality Classification and Enhancement

This project is focused on wall finish quality classification using image processing and machine learning. It enhances the texture of input images and classifies them as either "Good" or "Bad" based on certain metrics. The enhancement process uses techniques like CLAHE, bilateral filtering, sharpening, and edge detection. Additionally, the project exposes an API that allows users to upload images, enhance their texture, and classify them using a pre-trained deep learning model.

## Project Structure

The project consists of two main files:

1. **`final_c4.py`**:
    - Implements the image enhancement and texture analysis processes.
    - Calculates various texture metrics and classifies the image as "Good" or "Bad" based on pre-defined thresholds.

2. **`api.py`**:
    - Implements a Flask API that allows users to upload an image for processing and classification.
    - Uses the machine learning model to classify the image and responds with the classification result.

## Image Enhancement Process (Implemented in `final_c4.py`)

The `enhance_texture_single` function enhances the texture of an input image by following these steps:

### 1. Convert Image to Grayscale
The input image is converted to grayscale to simplify processing and focus on texture enhancement without color interference.

```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE is used to improve local contrast and make the details of the image clearer.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(gray_image)
```

### 3. Bilateral Filtering
This technique smoothens the image while preserving edges, reducing noise and improving visual quality.

```python
smoothed_image = cv2.bilateralFilter(equalized_image, d=9, sigmaColor=75, sigmaSpace=75)
```

### 4. Image Sharpening
A sharpening kernel is applied to the smoothed image to enhance edges and make features more distinct.

```python
sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)
```

### 5. Edge Detection (Canny)
The Canny edge detection algorithm is used to highlight the edges in the image.

```python
edges = cv2.Canny(sharpened_image, 50, 150)
```

### 6. Combine Enhanced Image and Edges
The final enhanced image is created by combining the sharpened image with the detected edges, giving it a more textured appearance.

```python
combined_image = cv2.addWeighted(sharpened_image, 0.8, edges, 0.2, 0)
```

The enhanced image is saved and displayed alongside the original image for comparison.

---

## Texture Metrics Computation (Implemented in `final_c4.py`)

The `compute_metrics` function calculates various metrics to assess the texture quality of the enhanced image:

1. **Edge Count**: The number of edge pixels detected using the Canny algorithm.
2. **Contrast**: The contrast in the image, calculated using Grey Level Co-occurrence Matrix (GLCM).
3. **Correlation**: The correlation between pixel intensities, calculated using GLCM.
4. **Energy**: The energy of the texture, calculated using GLCM.
5. **Homogeneity**: The homogeneity of the image, calculated using GLCM.
6. **Entropy**: The entropy of the image, calculated using its histogram.
7. **Mean and Standard Deviation**: The mean and standard deviation of pixel intensities.

```python
glcm = greycomatrix(image, [1], [0], symmetric=True, normed=True)
contrast = greycoprops(glcm, 'contrast')[0, 0]
correlation = greycoprops(glcm, 'correlation')[0, 0]
energy = greycoprops(glcm, 'energy')[0, 0]
homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
```

The metrics are returned as a dictionary for further analysis.

---

## Quality Classification (Implemented in `final_c4.py`)

The `classify_quality` function classifies the quality of the wall finish based on the computed metrics and predefined thresholds:

```python
if (metrics["edge_count"] < thresholds["edge_count"] and
    metrics["contrast"] < thresholds["contrast"] and
    metrics["correlation"] > thresholds["correlation"] and
    metrics["energy"] > thresholds["energy"] and
    metrics["homogeneity"] > thresholds["homogeneity"] and
    metrics["entropy"] < thresholds["entropy"]):
    return "Good"
```

- If the image passes the thresholds, it is classified as "Good".
- Otherwise, it is classified as "Bad".

---

## API Overview (Implemented in `api.py`)

The Flask application provides a RESTful API that allows users to upload images for texture enhancement and classification.

### Steps for Image Processing:

1. **Upload Image**: Users can send a `POST` request to `/predict` with the image file attached.
   
2. **Enhance Image**: The uploaded image is passed through the `enhance_texture_single` function for enhancement.

3. **Classify Image**: The enhanced image is then classified using the pre-trained model in the `predict` function.

4. **Return Classification**: The API returns a message indicating whether the image is classified as "Good" or "Bad".

### API Endpoint: `/predict`

- **Method**: `POST`
- **Input**: An image file (`image`) sent as form-data.
- **Output**: A JSON response containing the classification message.

#### Example cURL Request:

```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

#### Example Response:

```json
{
  "message": "The image is classified as good"
}
```

---

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- Numpy (`numpy`)
- Matplotlib (`matplotlib`)
- Scikit-Image (`scikit-image`)
- TensorFlow (`tensorflow`)
- Flask (`flask`)
- Flask-CORS (`flask-cors`)

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run the API

1. Clone the repository to your local machine:

```bash
git clone [<repository_url>](https://github.com/Srujan-rai/wall_texture_regcognition.git)
cd wall_texture_regcognition
```

2. Run the Flask API:

```bash
python api.py
```

The API will start running at `http://localhost:5000`.

---

## Conclusion

This project provides an image processing pipeline for enhancing wall texture and classifying it as "Good" or "Bad". The enhancement process improves the image's visual quality, while the classification is done based on both handcrafted texture metrics and a machine learning model. The Flask API enables easy integration with other systems by providing a simple interface to upload and process images.

