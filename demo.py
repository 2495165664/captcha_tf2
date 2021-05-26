import make_data
import cv2

image, label = make_data.make()


cv2.imshow("1", image[0])
cv2.waitKey(0)