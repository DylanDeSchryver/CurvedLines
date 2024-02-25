import cv2
from lineDetect import contourDetection
cap = cv2.VideoCapture(1)

while True:
    contourDetection(cap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
