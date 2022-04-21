import utils.HandTrackingModule as htm
import cv2
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities ,IAudioEndpointVolume
import pyautogui

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))

detector = htm.HandDetector(maxHands = 2,detectionCon=0.75)


cap_cam =cv2.VideoCapture(0)
cap_video=cv2.VideoCapture('video.mp4')
w=int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))

total_frames=int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
#print(total_frames)
ret,video_img=cap_video.read()

while cap_cam.isOpened():#카메라가 열려 있으면
    cam_ret,cam_img=cap_cam.read()
    cam_img = cv2.flip(cam_img, 1)
    #video_ret, video_img = cap_video.read()

    if not cam_ret:
        break
    hands,cam_img=detector.findHands(cam_img,flipType=False)
    if hands:
        lm_list=hands[0]['lmList']
        fingers=detector.fingersUp(hands[0])
        print(fingers)
        length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)
        if fingers == [0,0,0,0,0]:
            pass
        else:
            if length <50:
                rel_x = lm_list[4][0] / w
                if rel_x>1 : rel_x=1
                elif rel_x<0 : rel_x=0
                print(68*rel_x-68)
                volume.SetMasterVolumeLevel(65*rel_x-65, None)#0:max~ -65:0



    cv2.imshow('cam',cam_img)

    if cv2.waitKey(1) == ord('q'):
        break