import cv2
import pytesseract

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hendr\Project\Tesseract-OCR\tesseract.exe' # Path for my system , varies from system to system.

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set a lower resolution for the video feed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

# Counter to limit OCR frequency
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame counter
    frame_count += 1

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]

    # Perform OCR on every 5th frame
    if frame_count % 15 == 0:
        text = pytesseract.image_to_string(thresh,lang='eng')  # Perform OCR on the binary image
        print(text)  # Display the text in the console

    # Display the resulting frame (optional)
    cv2.imshow('frame', frame)
    cv2.imshow('Grayscale Frame', gray)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
