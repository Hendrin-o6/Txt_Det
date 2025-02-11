import cv2
import requests
import numpy as np

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Every 15th frame, send the image to the server for text extraction
    if frame_count % 15 == 0:
        _, img_encoded = cv2.imencode('.jpg', gray_image)
        response = requests.post(
            'http://127.0.0.1:5000/extract-text',
            files={'image': img_encoded.tobytes()}
        )
        text = response.json().get('text', '')
        # Print the text to the terminal
        print(text)

    # Draw the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frames
    cv2.imshow('Frame with Text', frame)
    cv2.imshow('Gray Frame', gray_image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
