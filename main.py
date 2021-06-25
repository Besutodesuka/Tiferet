import cv2
import handtracker as ht
import numpy as np
import autopy
import pyautogui as rt
import time
import pandas as pd

if __name__ == '__main__':
    debug = True
    past = 0
    # detector
    hand_detector = ht.handDetector(maxHands=1, detectCon=0.6)
    # mouse constants
    sensitivity = 4  # use to reduce the frame so it will improve the quality of mose
    currentlx, currently, previouslx, previously = 0, 0, 0, 0
    # get web cam
    cam_w, cam_h = 640, 480
    cam = cv2.VideoCapture(0)
    cam.set(3, cam_w)
    cam.set(4, cam_h)
    wScreen, hScreen = autopy.screen.size()
    frame_reduction = cam_h * 40 // 100
    overlay = False
    # fps getter
    fps = 1
    now = 0
    LeftClick = False
    RightClick = False
    # mode controller
    mode = ['none', 'mouse', 'gesture']
    mode_idx = 0
    mode_counters = 3
    mode_pointer = 0
    # gesture constants
    gesturelag = 0.1
    FPS = []
    while True:
        # get image
        success, img = cam.read()
        # get landmark
        img = hand_detector.get_landmark(img, draw=debug)
        positions, bbox = hand_detector.find_position(img, draw=debug)
        if len(positions) != 0:
            xmouse, ymouse = positions[9][1:]
            ymouse += 60
            cv2.circle(img, (xmouse, ymouse), 5, (255, 0, 255), cv2.FILLED)
            finger = hand_detector.get_fingeron()
            # pad frame
            #FPS.append(fps)
            if len(finger) != 0:
                # gesture condition
                if finger == [1, 1, 1, 1, 1]:
                    mode_pointer += 1/fps
                    if mode_pointer >= (mode_counters-1.5):
                        # minus 1.5 to balance time lost in computing lag
                        mode_idx = (mode_idx + 1) % len(mode)
                        mode_pointer = 0
                        if mode[mode_idx] != 'mouse':
                            RightClick = False
                            LeftClick = False
                if finger != [1, 1, 1, 1, 1]:
                    mode_pointer = 0
                if finger[0] + finger[3] + finger[4] == 0 and mode[mode_idx] == 'mouse':
                    # distance, _ = hand_detector.get_distance(img, 8, 12)
                    x = np.interp(xmouse, (frame_reduction, cam_w - frame_reduction), (0, wScreen))
                    y = np.interp(ymouse, (frame_reduction, cam_h - frame_reduction), (0, hScreen))
                    # smooth the value
                    currentlx = min(previouslx + (x - previouslx) / sensitivity, wScreen-1)
                    currently = min(previously + (y - previously) / sensitivity, hScreen-1)
                    # move the mouse
                    autopy.mouse.move((wScreen - currentlx), currently)
                    previouslx, previously = currentlx, currently
                    # xmouse, ymouse
                    # click mode
                    Lclick = finger == [0, 0, 1, 0, 0]
                    Rclick = finger == [0, 1, 0, 0, 0]
                    if Lclick == True and LeftClick == False:
                        rt.mouseDown(button=rt.LEFT)
                        LeftClick = True
                        print('Ldown')
                    if Lclick == False and LeftClick == True:
                        rt.mouseUp(button=rt.LEFT)
                        LeftClick = False
                        print('Lup')
                    if Rclick == True and RightClick == False:
                        rt.mouseDown(button=rt.RIGHT)
                        RightClick = True
                        print('Rdown')
                    if Rclick == False and RightClick == True:
                        rt.mouseUp(button=rt.RIGHT)
                        RightClick = False
                        print('Rup')
                elif mode[mode_idx] == 'gesture':
                    if finger == [0, 0, 0, 0, 0]:
                        rt.keyDown('win')
                        rt.press('d')
                        rt.keyUp('win')
                        print('to desktop')
                        time.sleep(gesturelag)
                    if finger == [0, 1, 1, 0, 0]:
                        rt.press('f5')
                        print('refresh')
                        time.sleep(gesturelag)
                    if finger == [1, 1, 0, 0, 1]:
                        rt.keyDown('win')
                        rt.press('prtsc')
                        rt.keyUp('win')
                        print('print screen')
                        FPS.append(fps)
                        time.sleep(gesturelag)
                elif mode == 'none':
                    pass
            else:
                RightClick = False
                LeftClick = False
                rt.mouseUp(button=rt.RIGHT)
                rt.mouseUp(button=rt.LEFT)
        # display
        if debug == True:
            # fps get
            now = time.time()
            fps = 1 / (now - past)
            past = now
            cv2.rectangle(img, (frame_reduction, frame_reduction), (cam_w - frame_reduction, cam_h - frame_reduction),
                          (255, 0, 255), 2)
            cv2.putText(img, 'FPS : {}'.format(int(fps)), (5, 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 2)
            cv2.putText(img, 'mode : {}'.format(mode[mode_idx]), (5, 70), cv2.FONT_ITALIC, 1, (255, 0, 255), 2)
            cv2.imshow('detector', img)
        cv2.waitKey(1)
        if cv2.getWindowProperty('detector', cv2.WND_PROP_VISIBLE) <= 0:
            print('closed')
            cv2.destroyAllWindows()
            data = pd.DataFrame({'FPS': FPS})
            data.to_csv('fpslog.csv')
            break





