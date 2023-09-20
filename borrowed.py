import cv2
import numpy as np

resolution_x = 640
resolution_y = 480

camera = cv2.VideoCapture(0)
# Start the OpenCV window thread
cv2.startWindowThread ()



counter = 0

while True:
    counter += 1

# Capture a frame from the camera
    ret, frame = camera.read()
    if not ret or counter > 5:
        break

    
# Display the camera feed with detection results    
    #we blur to filter out noise
    blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
        # Convert the frame to HSV color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    # Define the range of colors for detecting the tennis ball (you may need to adjust these values)
    lower_color = np.array([15, 100, 20])
    upper_color = np.array([62, 255, 255])
    # Create a mask to threshold the frame
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE )
    # Check if any detected contour meets the size criteria
    for index, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # Centroid (cx, cy) of the contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print(f"Contour centroid at: ({cx}, {cy})")
        else:
            print("Contour area is zero.")
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # contourArea calculates the approximate size of the found contour
        area = cv2.contourArea(contour)
        if area > 50: # Adjust the size threshold for the detected tennis ball
            print("Detected a tennis ball above the size threshold." )
            cv2.drawContours(frame , contours, index, (0,255,0), 1)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        else:
            print("Detected something yellow, but it's below the size threshold.")
    # Display the camera feed with detection results
    cv2.imwrite(f'mask-{counter}.png' , mask)
    print(len(contours))
    # for index, c in enumerate(contours):
    #     cv2.drawContours(frame , contours, index, (0,255,0), 1)
    cv2.imwrite(f'image-{counter}.png' , frame)
    # print(mask)
    cv2.waitKey(1)


# Release the camera and close the OpenCV window
camera.release()
cv2.destroyAllWindows ()