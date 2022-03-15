import cv2
import numpy as np
import math

video = cv2.VideoCapture('NAC_VIDEO.mp4')

while(1):
    ret, frame = video.read()

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame_lower_hsv = np.array([25, 80, 110])  
    frame_upper_hsv = np.array([170, 255, 255])

    mask_hsv = cv2.inRange(frame_hsv, frame_lower_hsv, frame_upper_hsv)
    res = cv2.bitwise_and(frame,frame, mask_hsv)
    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"

    def get_contour_areas(contours):

        all_areas= []

        for cnt in contours:
            area= cv2.contourArea(cnt)
            all_areas.append(area)

        return all_areas

    sorted_contours= sorted(contornos, key=cv2.contourArea, reverse= True)

    largest_item= sorted_contours[0], sorted_contours[1]

    cv2.drawContours(contornos_img, largest_item, -1, [0, 255, 0], 5)

    cnt = largest_item

    M = cv2.moments(cnt[0]), cv2.moments(cnt[1])

    cx = int(M[0]['m10']/M[0]['m00']),int(M[1]['m10']/M[1]['m00'])
    cy = int(M[0]['m01']/M[0]['m00']),int(M[1]['m01']/M[1]['m00'])


    size = 20
    color = (128,128,0)


    cv2.line(contornos_img,(cx[0] - size,cy[0]),(cx[0] + size,cy[0]),color,5)
    cv2.line(contornos_img,(cx[0],cy[0] - size),(cx[0], cy[0] + size),color,5)
    cv2.line(contornos_img,(cx[1] - size,cy[1]),(cx[1] + size,cy[1]),color,5)
    cv2.line(contornos_img,(cx[1],cy[1] - size),(cx[1], cy[1] + size),color,5)

    # Para escrever vamos definir uma fonte 

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = cy , cx
    origem = (0,50)

    start_point = (cx[0], cy[0])
    end_point = (cx[1], cy[1])
    line = cv2.line(contornos_img,start_point, end_point, color, 5)

    myradians = math.atan2(cy[0]-cy[1], cx[0]-cx[1])
    mydegrees = math.degrees(myradians)
    print(mydegrees)

    cv2.putText(contornos_img, str(text), origem, font,1,(200,50,0),2,cv2.LINE_AA)

    cv2.imshow('frame',contornos_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()