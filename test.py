import cv2 
import numpy as np 

camera = cv2.VideoCapture(0)

def get_color_area(image_path, color):
    
    image = cv2.imread(image_path)
    mask = cv2.inRange(image, color, color)
    area = cv2.countNonZero(mask)
    
    return area

def get_direction():
    
    while camera.isOpened():
        # Capture a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break

        blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        
        lower_color = np.array([30, 100, 100])
        upper_color = np.array([62, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust the size threshold for the detected tennis ball
                #print("Detected a tennis ball above the size threshold.")
                cv2.imwrite('detected_tennis_ball.jpg', frame)
        
        img = cv2.imread("detected_tennis_ball.jpg")

        cv2.drawContours(img, contours, -1, (255,0,127), cv2.FILLED)
        cv2.imwrite('Contours.jpg', img)

        height, width, _ = img.shape

        column_width = width // 5

        # Split the image into 5 columns
        columns = [img[:, i*column_width:(i+1)*column_width] for i in range(5)]

        # Save each column as a separate image
        for idx, col in enumerate(columns):
            cv2.imwrite(f'column_{idx+1}.jpg', col)
        
        max_area = 0
        max_area_column = None
        for idx in range(5):
            column_image_path = f'column_{idx+1}.jpg'
            area = get_color_area(column_image_path, (255, 0, 127))
            #print(f'Area of color in {column_image_path}: {area}')
            if area > max_area:
                max_area = area
                max_area_column = idx+1
        
        if max_area_column is not None:
            print(f'Column image with the most area of color: {max_area_column}')
        
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        
    
    return max_area_column

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_direction()