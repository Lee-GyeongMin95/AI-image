import utils.HandTrackingModule as htm
import cv2
import numpy as np
import pyautogui
######### 손모양으로 window 기본 오디오 조절하기
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities ,IAudioEndpointVolume

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
###############################################
cap = cv2.VideoCapture(0)
detector = htm.HandDetector(maxHands = 1,detectionCon=0.75)

file=np.genfromtxt('data3.csv',delimiter=',')# 파일을 읽어온다.  delimiter 는 split  랑 역할이 같다.

angle=file[:,:-1].astype(np.float32)#0번인덱스 부터 마지막 인덱스(-1) 전까지 잘라라.   64비트로 가져오면 에러가 나서 변경한 후 가져온다.
label=file[:,-1].astype(np.float32)#마지막 인덱스(-1)만 가져와라

knn=cv2.ml.KNearest_create()#knn모델을 초기화. ml은 머신러닝 약자
knn.train(angle,cv2.ml.ROW_SAMPLE,label)#knn 학습

rps_gesture = {0:'rock', 1:'scissors', 2:'paper'}

while cap.isOpened():
    cam_ret, cam_img = cap.read()
    cam_img = cv2.flip(cam_img,1)
    hands, cam_img = detector.findHands(cam_img, flipType=True)
    ######## 소리 조절
    # if hands:
    #     lm_list = hands[0]['lmList']
    #     fingers = detector.fingersUp(hands[0])
    #     print(fingers)
    #     length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)
    #     h,w,c = cam_img.shape
    #     if length < 50:
    #         rel_x = lm_list[4][0] / w
    #         if rel_x > 1:
    #             rel_x = 1
    #         elif rel_x < 0:
    #             rel_x = 0
    #         print(68 * rel_x - 68)
    #         volume.SetMasterVolumeLevel(65 * rel_x - 65, None)  # 0:max~ -65:0
    ######## 방향키
    if hands:
        lm_list = hands[0]['lmList']
        lm_list = np.array(lm_list)## 벡터 연산 쉽게 할라고 배열로 만든다.
        v1 = lm_list[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = lm_list[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

        v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)
        # 유닛벡터 구하기 벡터/벡터의 길이/// expand_dim 축 추가
        ##np.linalg.norm(v, axis=1) 만 하면 (20,3), (20,) 로 나온다. 축이 없어서 에러 뜬다.
        ## 그래서 expand_dim 로 축을 추가한다. 뒤에 축을 추가해야해서 axis  = -1 을 한다.
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                    :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
        ## arccos 아크코사인    ,   einsum 행렬 연산속도 젤 빠름     ,  각도는 라디안으로 나옴
        angle = np.degrees(angle)  # Convert radian to degree
        angle = np.expand_dims(angle.astype(np.float32),axis=0)
        # float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다. 모델을 학습할때는 필요없다.
        _, results, _, _ = knn.findNearest(angle, 3)  # statue,result,인접값,거리
        # print(results)
        idx = int(results[0][0])
        gesture_name = rps_gesture[idx]

        print(gesture_name)
        cv2.putText(cam_img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=2)

    if not cam_ret:
        break
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('image view', cam_img)

