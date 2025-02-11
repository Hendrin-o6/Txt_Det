import cv2
import pytesseract

# Set the correct path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\hendr\Project\Tesseract-OCR\tesseract.exe'

cap = cv2.VideoCapture(0)
frame_count = 0
text = ""  # Initialize text variable

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5,5),0)
    
    # Every 15th frame, extract text
    if frame_count % 15 == 0:
        text = pytesseract.image_to_string(gray_image, lang='eng')
        print(text)


    # Draw extracted text on the frame
    cv2.putText(frame, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the processed frames
    cv2.imshow('Frame with Text', frame)
    cv2.imshow('Gray Frame', gray_image)

    # Press 'q' to exit
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()