import sys
import cv2
import numpy as np


image_file = "input.png"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Перевод изображения в чб
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)#Увеличение изображения

#TODO: Перевод исходного изображения в ЧБ,для корректного распознования блоков
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#h_min = np.array((255, 255, 255), np.uint8)
#h_max = np.array((0, 0, 0), np.uint8)
#img2 = cv2.inRange(hsv, h_min, h_max)
#cv2.imshow("kek", img_erode)

# Нахождение блоков(контуров)
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
output = img.copy()
letters = []
for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
       cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
       letter_crop = gray[y:y + h, x:x + w]  # print(letter_crop.shape)

       

#cv2.imshow("Input", img)
#cv2.imshow("Enlarged", img_erode)
cv2.imshow("Output", output)

#Запись блоков в отдельные файлы


#Распознование линий
img = cv2.imread("input.png",0)
#Create default Fast Line Detector (FSD)
fld = cv2.ximgproc.createFastLineDetector(_length_threshold=37)
#Detect lines in the image
lines = fld.detect(img)
#Draw detected lines in the image
drawn_img = fld.drawSegments(img,lines)
#Show image
cv2.imshow("FLD", drawn_img)
cv2.waitKey(0)
