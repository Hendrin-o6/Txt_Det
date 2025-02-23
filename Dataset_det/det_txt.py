import cv2
import torch
import numpy as np
import os
import easyocr
import matplotlib.pyplot as plt

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Directory containing images
image_dir = r'c:\Users\hendr\Downloads\Text.v3i.yolov5pytorch\valid\images'
output_dir = r'C:\Users\hendr\Downloads\Output'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to preprocess image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive histogram equalization (CLAHE) to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

# Iterate over images in the directory
for image_name in os.listdir(image_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(image_dir, image_name)
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        try:
            # Perform inference with updated autocast
            with torch.amp.autocast('cuda'):
                results = model(image)
        
            # Initialize a list to hold detected text
            detected_texts = []

            # Draw bounding boxes and labels on the image
            for detection in results.xyxy[0].numpy():
                x1, y1, x2, y2, conf, cls = detection
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the detected text region
                text_region = image[int(y1):int(y2), int(x1):int(x2)]
                # Preprocess the text region
                preprocessed_text_region = preprocess_image(text_region)
                # Perform OCR using EasyOCR with fine-tuned settings
                ocr_results = reader.readtext(preprocessed_text_region, detail=0, batch_size=8)

                # Draw detected text on the image
                for text in ocr_results:
                    detected_texts.append(text)
                    cv2.putText(image, text, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save the output image with bounding boxes
            output_image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_image_path, image)

            # Display the image with bounding boxes and detected text using Matplotlib
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'YOLO Text Detection - {image_name}')
            plt.axis('off')  # Hide axes for better visualization
            plt.show()

            # Print detected text in the terminal
            print(f"Detected Text in {image_name}:")
            for idx, text in enumerate(detected_texts):
                print(f"Text {idx + 1}: {text}")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
