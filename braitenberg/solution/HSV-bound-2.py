import cv2
import numpy as np
import sys
import time
import os


def nothing(x):
    pass

os.chdir("../samples/many-duckies/")

# # Open the camera
# cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("image")

# frame0 = dcu.image_cv_from_jpg_fn(sys.argv[1])
lastL = np.array([171, 140, 0])
lastU = np.array([179, 200, 255])
Gimg_no = 1

# create trackbars for color change
cv2.createTrackbar("1_lowH", "image", lastL[0], 179, nothing)
cv2.createTrackbar("2_highH", "image", lastU[0], 179, nothing)

cv2.createTrackbar("3_lowS", "image", lastL[1], 255, nothing)
cv2.createTrackbar("4_highS", "image", lastU[1], 255, nothing)

cv2.createTrackbar("5_lowV", "image", lastL[2], 255, nothing)
cv2.createTrackbar("6_highV", "image", lastU[2], 255, nothing)
cv2.createTrackbar("7_img_no", "image", Gimg_no, 19, nothing)



while True:
    Gimg_no = cv2.getTrackbarPos("7_img_no", "image")
    frame = cv2.imread("many-duckies-%02d.jpg" % Gimg_no)
    # get current positions of the trackbars
    AilowH = cv2.getTrackbarPos("1_lowH", "image")
    BihighH = cv2.getTrackbarPos("2_highH", "image")
    CilowS = cv2.getTrackbarPos("3_lowS", "image")
    DihighS = cv2.getTrackbarPos("4_highS", "image")
    EilowV = cv2.getTrackbarPos("5_lowV", "image")
    FihighV = cv2.getTrackbarPos("6_highV", "image")

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([AilowH, CilowS, EilowV])
    higher_hsv = np.array([BihighH, DihighS, FihighV])
    if not np.allclose(lastL, lower_hsv) or not np.allclose(lastU, higher_hsv):

        print(f"lower {lower_hsv} upper {higher_hsv}")
        lastL = lower_hsv
        lastU = higher_hsv

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("image", frame)
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
