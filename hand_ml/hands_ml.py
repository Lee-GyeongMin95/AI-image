import utils.HandTrackingModule as htm
import cv2
import numpy as np
import pyautogui
import pickle

rps_gesture = {0:'rock', 1:'scissors', 2:'paper'}


detector = htm.HandDetector(maxHands = 2,detectionCon=0.75)

file=np.genfromtxt('count.csv',delimiter=',')# 파일을 읽어온다

angle=file[:,:-1].astype(np.float32)#0번인덱스 부터 마지막 인덱스(-1) 전까지 잘라라
label=file[:,-1].astype(np.float32)#마지막 인덱스(-1)만 가져와라


knn=cv2.ml.KNearest_create()#knn모델을 초기화
knn.train(angle,cv2.ml.ROW_SAMPLE,label)#knn 학습



cap_cam =cv2.VideoCapture(0)
cap_video=cv2.VideoCapture('video.mp4')
w=int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))

total_frames=int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
#print(total_frames)
ret,video_img=cap_video.read()
flag=1
while cap_cam.isOpened():#카메라가 열려 있으면
    cam_ret,cam_img=cap_cam.read()
    cam_img = cv2.flip(cam_img, 1)
    #video_ret, video_img = cap_video.read()

    if not cam_ret:
        break
    hands,cam_img=detector.findHands(cam_img,flipType=False)
    if hands:
        lm_list=hands[0]['lmList']
        lm_list=np.array(lm_list)

        v1 = lm_list[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = lm_list[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

        v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)  # 유닛벡터 구하기 벡터/벡터의 길이

        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                    :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
        angle = np.degrees(angle)  # Convert radian to degree
        angle = np.expand_dims(angle.astype(np.float32),
                               axis=0)  # float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다.
        _, results, _, _ = knn.findNearest(angle, 3)  # statue,result,인접값,거리
        # print(results)
        idx = int(results[0][0])
        gesture_name = rps_gesture[idx]

        print(gesture_name)
        cv2.putText(cam_img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=2)
        # if  idx==1:pyautogui.press('left')
        # elif idx==2:pyautogui.press('right')



        #elif idx==3:pyautogui.press('right')

        # if idx==1 and flag==0:
        #     pyautogui.press('left')
        #     flag=1
        # elif idx==2 and flag==0:
        #     pyautogui.press('right')
        #     flag = 1
        # elif idx==0 and flag==1:
        #     flag=0





    cv2.imshow('cam',cam_img)
    idx = 3

    if cv2.waitKey(1) == ord('q'):
        break