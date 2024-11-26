import cv2 
import numpy as np 


#camera = cv2.VideoCapture(0)


def get_color_area(image_path, color):
    
    image = cv2.imread(image_path)
    mask = cv2.inRange(image, color, color)
    area = cv2.countNonZero(mask)
    
    return area

#Takes a picutre
def take_picture(camera):
    if camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            return

        blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        return hsv, frame
    else:
        return None, None

#Image processing
def get_image(hsv, frame, lower_color, upper_color):
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust the size threshold for the detected tennis ball
            cv2.imwrite('detected_color.jpg', frame)
    
    img = cv2.imread("detected_color.jpg")

    cv2.drawContours(img, contours, -1, (255,0,127), cv2.FILLED)

    # Save the image with contours back to the file
    cv2.imwrite('detected_color.jpg', img)

    return get_area(img), get_direction(img)

def get_direction(img):
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
    
    #if max_area_column is not None:
       #print(f'Column image with the most area of color: {max_area_column}')
    return max_area_column

def get_area(img):
    area = get_color_area("detected_color.jpg", (255, 0, 127))
    return area

