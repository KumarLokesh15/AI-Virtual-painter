import cv2
import mediapipe as mp
import time
import math as m

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=0, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=1):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img

    def findPosition(self, img, handNo=0, draw=1):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for Id, Lm in enumerate(myHand.landmark):
                # print(Id, '\n', Lm)
                h,w,c = img.shape
                cx, cy = int(Lm.x*(w)), int(Lm.y*(h))
                xList.append(cx)
                yList.append(cy)
                # print(Id, cx, cy)
                self.lmList.append([Id,cx,cy])
                # if Id == 0:
                if draw:
                    cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-20,bbox[1]-20), (bbox[2]+20,bbox[3]+20) , (0,0,255), 2)
        return self.lmList, bbox

    def fingersUp(self,img):
        self.tipIds = [4,8,12,16,20]
        fingers = []
        if len(self.lmList)!=0:
            #Thumb
            #Right Hand Thumb
            if self.lmList[0][1] > self.lmList[2][1]:
                cv2.putText(img,'Right Hand',(1000,700), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            #Left Hand Thumb
            elif self.lmList[0][1] < self.lmList[2][1]:
                cv2.putText(img,'Left Hand',(1000,700), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
            #4fingers
            for Id in range(1,5):
                if self.lmList[self.tipIds[Id]][2] < self.lmList[self.tipIds[Id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=1):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255,0,0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)
            cv2.circle(img, (cx, cy), 10, (255,0,0), cv2.FILLED)
            
        length = m.hypot(x2-x1,y2-y1)
        return length, img, [x1,y1,x2,y2,cx,cy]




def main():
    prevTime = 0
    CurTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while 1:
        _,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        curTime = time.time()
        fps = 1/(curTime-prevTime)
        prevTime = curTime

        cv2.putText(img,'FPS : '+str(int(fps)),(10,700), cv2.FONT_HERSHEY_PLAIN, 3, (255,244,0), 3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()