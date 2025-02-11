from flask import Flask, request, jsonify
import pytesseract
import cv2
import numpy as np

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hendr\Project\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Text Extraction Server!'

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return 'No image file provided', 400

    image_file = request.files['image'].read()
    np_img = np.frombuffer(image_file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)

    print(f"Received text: {text}")
    return jsonify({'text': text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
