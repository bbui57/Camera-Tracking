import cv2 as cv
import numpy as np

#Initialize webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print ("Cannot open camera")
    exit()

#read the first two frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    
    #get the difference between the two frames
    diff = cv.absdiff(frame1, frame2)

    #make the difference grayscale, helps with processing
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY) 

    #blur the image to reduce noise
    blur = cv.GaussianBlur(gray, (5,5), 0)

    #apply a binary threshold to the image
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)

    #dilate the image to fill in holes
    dilated = cv.dilate(thresh, None, iterations=3)

    #find contours in the dilated image
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) < 500:
            continue
        #get the bounding box of the contour
        (x, y, w, h) = cv.boundingRect(contour)
        #draw the bounding box
        cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Camera', frame1)

    #update the frames
    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    #if the user presses 'q' the camera will close
    if cv.waitKey(1) == ord('q'):
        break

#release the camera and close the window
cap.release()
cv.destroyAllWindows()