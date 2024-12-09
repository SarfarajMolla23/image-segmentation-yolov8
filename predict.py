from ultralytics import YOLO
import cv2
import numpy as np
import os

# Paths
model_path = r'C:\Users\sarfa\Documents\PythonProject\image-segmentation-yolov8\runs\segment\train4\weights\last.pt'
image_path = r'C:\Users\sarfa\Documents\PythonProject\image-segmentation-yolov8\data\images\val\duck89.jpeg'

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

H, W, _ = img.shape

# Load YOLO model
model = YOLO(model_path)

# Run inference
results = model(img)

# Create output directory
output_dir = "./output_masks"
os.makedirs(output_dir, exist_ok=True)

# Save masks
for i, result in enumerate(results):
    if result.masks is not None:
        for j, mask in enumerate(result.masks.data):
            # Convert mask to numpy array and scale to 255
            mask = (mask.numpy() * 255).astype(np.uint8)

            # Resize mask to match original image dimensions
            mask = cv2.resize(mask, (W, H))

            # Save the mask with a unique filename
            output_path = os.path.join(output_dir, f"mask_{i}_{j}.png")
            cv2.imwrite(output_path, mask)
            print(f"Saved: {output_path}")
    else:
        print(f"No masks detected in result {i}.")
