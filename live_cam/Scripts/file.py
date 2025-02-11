import cv2
import pytesseract
import os

# Set the correct path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hendr\Project\Tesseract-OCR\tesseract.exe'

# Define constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_POSITION = (10, 30)
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
THICKNESS = 1
IMAGE_DIR = 'C:\Users\hendr\OneDrive\Pictures\Screenshots'  # Set this to your image directory

# Function to process images and detect text
def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Extract text using Tesseract
    text = pytesseract.image_to_string(gray_image, lang='eng')
    print(f'Text in {image_path}: {text}')

    # Draw extracted text on the image
    cv2.putText(image, text, TEXT_POSITION, FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

    # Display the processed image
    cv2.imshow(f'Image with Text: {os.path.basename(image_path)}', image)
    cv2.imshow(f'Gray Image: {os.path.basename(image_path)}', gray_image)

# Read images from the directory and process them
for image_file in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_file)
    if os.path.isfile(image_path):
        process_image(image_path)
        cv2.waitKey(0)  # Press any key to display the next image

cv2.destroyAllWindows()
