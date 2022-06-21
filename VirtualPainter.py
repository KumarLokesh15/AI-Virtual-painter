import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 20
eraserThickness = 100

pTime = 0
cTime = 0

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)

overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))

header = overlayList[0]

drawColor = (255,255,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

xp, yp = 0,0
imgCanvas = np.zeros((720,1280,3), np.uint8)
flag = 0

detector = htm.handDetector(detectionConfidence=0.85)

while 1:
    # 1. import image
    _, img = cap.read()
    img = cv2.flip(img,1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList,_ = detector.findPosition(img, draw=1)
    if len(lmList)!=0:
        # print(lmList)
        #tip of index & middle finger 
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
    # 3. Check which fingers are up
        fingers = detector.fingersUp(img)
        # print(fingers)
    # 4. if selection mode ~ 2 fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
            print("Selection Mode")
            #Checking for the click
            if y1<125:
                flag = 1
                if 28<x1<98:
                    header = overlayList[1]
                    drawColor = (0,0,255)
                    cv2.putText(img,'Red',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 120<x1<192:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                    cv2.putText(img,'Green',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 218<x1<290:
                    header = overlayList[3]
                    drawColor = (255,0,0)
                    cv2.putText(img,'Blue',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 310<x1<389:
                    header = overlayList[4]
                    drawColor = (0,255,255)
                    cv2.putText(img,'yellow',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 415<x1<494:
                    header = overlayList[5]
                    drawColor = (255,255,0)
                    cv2.putText(img,'Cyan',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 516<x1<599:
                    header = overlayList[6]
                    drawColor = (255,0,255)
                    cv2.putText(img,'Pink',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                # if 622<x1<704:
                #     header = overlayList[7]
                #     drawColor = (0,0,0)
                #     cv2.putText(img,'Black',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                # if 724<x1<810:
                #     header = overlayList[8]
                #     drawColor = (255,255,255)
                #     cv2.putText(img,'White',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                if 1073<x1<1267:
                    header = overlayList[-1]
                    drawColor = (0,0,0)
                    cv2.putText(img,'Eraser',(10,700), cv2.FONT_HERSHEY_PLAIN, 3, drawColor, 5)
                # cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)


    # 5. if drawing mode ~ index finger is up
        if fingers[1] and fingers[2]==0 and flag == 1:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print('Drawing Mode')
            if xp==0 and yp==0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    cv2.putText(img,'FPS : '+str(int(fps)),(10,700), cv2.FONT_HERSHEY_PLAIN, 3, (255,244,0), 3)

    #6. setting the header image
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow('image',img)
    # cv2.imshow('image',imgCanvas)
    key = cv2.waitKey(1)
    if key == ord('q') & 0xFF:
        break
