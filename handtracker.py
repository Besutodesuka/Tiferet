'''
module for extract hands features
'''
import math
import mediapipe as mp
import cv2

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def get_landmark(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(detection.multi_hand_landmark)
        self.detection = self.hands.process(img_rgb)
        if self.detection.multi_hand_landmarks:
            for hand in self.detection.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand, self.mphands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True, label=False):
        x = []
        y = []
        self.landmark = []
        bbox = []
        if self.detection.multi_hand_landmarks:
            hand = self.detection.multi_hand_landmarks[handNo]
            height, width, c = img.shape

            for id, landmark in enumerate(hand.landmark):
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                x.append(cx)
                y.append(cy)
                self.landmark.append([id, cx, cy])
                if label:
                    cv2.putText(img, '{} {} {}'.format(id, cx, cy), (cx, cy), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            bbox = x_min, y_min, x_max, y_max
            if draw:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.landmark, bbox

    def get_distance(self, img, firstid, lastid, draw=True):
        if len(self.landmark) == 0:
            return None
        else:
            x1, y1 = self.landmark[firstid][1:]
            x2, y2 = self.landmark[lastid][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
            return length, [x1, y1, x2, y2, cx, cy]

    def get_fingeron(self):
        fingers = []
        tips_id = [4, 8, 12, 16, 20]
        # detect 4 finger
        if len(self.landmark) != 0:
            for id in tips_id:
                if id == 4:
                    # in [4, 8, 12]:
                    if self.landmark[id][1] > self.landmark[id - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    #
                    if self.landmark[id][2] < self.landmark[id - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        return fingers

