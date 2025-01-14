from PIL import Image

# Open the image
image_path = "model_creation/20240122_102205_jpg.rf.7e014af095b52a00598f271c2e9d819f.jpg"
image = Image.open(image_path)

# Get the mode
print(f"Image mode: {image.mode}")

# Mode explanations:
# 'L' → Grayscale
# 'RGB' → 3 channels
# 'RGBA' → 4 channels
