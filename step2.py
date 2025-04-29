import cv2 as cv
import numpy as np

img = cv.imread("cropped_images/coca.jpg")
cv.imshow("Image originale", img)

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

cv.waitKey(0)

h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
cv.waitKey(0)

# define range of blue color in HSV
lower_range = np.array([0, 200, 200])
upper_range = np.array([24, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(img_hsv, lower_range, upper_range)


res = cv.bitwise_and(img, img, mask=mask)
cv.imshow("Masque + img", res)

imgray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 1, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(res, contours, -1, (255, 255, 255), 3)

cv.imshow("Contours détectés", res)

cv.waitKey(0)
cv.destroyAllWindows()





