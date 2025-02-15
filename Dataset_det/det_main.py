import cv2
from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hendr\textdet\Tesseract-OCR\tesseract.exe'
 
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    threshold_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return threshold_image

def recognize_text(image):
    text = pytesseract.image_to_string(image)
    return text

def process_dataset(dataset_path):
    for image_file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_file)
        preprocessed_image = preprocess_image(image_path)
        text = recognize_text(preprocessed_image)
        print(f'Text from {image_file}:')
        print(text)
        print('---')

dataset_path = r'C:\Users\hendr\Downloads\Text.v3i.yolov5pytorch\valid\images'
process_dataset(dataset_path)
