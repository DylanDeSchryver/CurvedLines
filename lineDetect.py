import cv2
import numpy as np
from midline import midCalc


def contourDetection(cap):

    ret, frame = cap.read()

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Gaussion Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find Canny edges
    edged = cv2.Canny(blurred, 30, 200)

    r_x, r_y, r_width, r_height = 100, 100, 300, 300  # camera at home
    # r_x, r_y, r_width, r_height = 300, 100, 800, 500 #camera at school

    # Draw a rectangle around the ROI on the original frame
    cv2.rectangle(frame, (r_x, r_y), (r_x + r_width, r_y + r_height), (255, 0, 0), 2)

    # Apply the defined ROI on the processed edge image
    roi_edges = edged[r_y:r_y + r_height, r_x:r_x + r_width]

    # Finding Contours
    contours, hierarchy = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Ensure that contours are found before proceeding
    if len(contours) >= 2:
        # Sort contours by area to find the largest ones (2 curved lines)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        # Draw all contours within the ROI
        for contour in contours:
            contour += (r_x, r_y)  # Offset contour points to match ROI in the original frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 4)

    # draw midline
    try:
        cv2.polylines(frame, [np.array(midCalc(contours))], False, (0, 0, 255), 4)
    except cv2.error:
        pass

    # display frame

    return cv2.imshow('Contours', frame)
