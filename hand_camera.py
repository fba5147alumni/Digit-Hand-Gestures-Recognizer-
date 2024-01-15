import cv2
import mediapipe as mp
import numpy as np
import torch

class FingerDrawer():
    def __init__(self, net_actions):
        self.cap = cv2.VideoCapture(0) 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.canvas = np.zeros((300,300), dtype="uint8")
        self.net_actions = net_actions

    def hand_draw(self):
        while True:
            success, img = self.cap.read()
            img = cv2.resize(img,(300,300))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            lmList = []
            if results.multi_hand_landmarks: 
                handlandmark = results.multi_hand_landmarks[0]
                lm = handlandmark.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy]) 
            if lmList != []:
                p = lmList[0][1], lmList[0][2]
                cv2.line(self.canvas,p,p,255,15)
            cv2.imshow('Image',self.canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break 
            elif key == ord('c'):
                cv2.imwrite("Test_Canvas.png", self.canvas)
                self.canvas[0:300,0:300] = 0
            image = self.canvas[0:300,0:300]
            image = cv2.resize(image,(28,28)).reshape(1,1,28,28).astype('float32')/255
            result = self.net_actions.predict(torch.tensor(image))
            print("Prediction : ",result)
        self.cap.release()     
        cv2.destroyAllWindows()
