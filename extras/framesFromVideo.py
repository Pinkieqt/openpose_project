import os
import moviepy.editor as mp
import cv2


directory = "/home/pinkie/Downloads/"
directoryToSave = "/home/pinkie/Downloads/savedFramesFromVideo/"
fileName = "SILVER_1.mpeg"

video = cv2.VideoCapture(directory + fileName)

i = 0

while (video.isOpened()):
    ret, frame = video.read()
    if ret == False:
        break
    cv2.imwrite(directoryToSave + 'img' + str(i) + '.jpg',frame)
    print("Snímek číslo: " + str(i))
    i+=1
 
video.release()
cv2.destroyAllWindows()
