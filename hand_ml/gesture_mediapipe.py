import cv2
import mediapipe as mp
import numpy as np

gesture = {
    0:'0', 1:'1',2:'2',
}
rps_gesture = {0:'rock', 1:'paper', 9:'scissors'}

cap =cv2.VideoCapture(0)
#MediaPiipe Hands 손의 관절 위치를 인식할 수 있는 모델을 초기화 한다.
mp_hands= mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(
    max_num_hands=1,#최대 손의 인식 갯수
    min_detection_confidence=0.5,# 탐지 임계치
    min_tracking_confidence=0.5# 추적 임계치
)
file=np.genfromtxt('data3.csv',delimiter=',')# 파일을 읽어온다

angle=file[:,:-1].astype(np.float32)#0번인덱스 부터 마지막 인덱스(-1) 전까지 잘라라
label=file[:,-1].astype(np.float32)#마지막 인덱스(-1)만 가져와라


knn=cv2.ml.KNearest_create()#knn모델을 초기화
knn.train(angle,cv2.ml.ROW_SAMPLE,label)#knn 학습



print(file)

while cap.isOpened():#카메라가 열려 있으면
    ret,img=cap.read()# 카메라의 프레임을 한 프레임씩 읽는다.
    img=cv2.flip(img,1)
    input_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(input_img)# 웹캠 이미지에서 손의 위치 관절 위치를 탐지한다.
    if result.multi_hand_landmarks is not None:# 손이 인식 되면



        for res in result.multi_hand_landmarks:#인식되 손의 갯수만큼 포문을 돌면서
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)# 그림을 그린다.
            joint=np.zeros((21,3))

            for j,lm in enumerate(res.landmark):#21개의 랜드 마크가 들어있는데 한점씩 반복문을 사용하여 처리한다.
                joint[j]=[lm.x, lm.y, lm.z]


            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v=v2-v1#(20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

            v=v/np.expand_dims(np.linalg.norm(v,axis=1),axis=-1)#유닛벡터 구하기 벡터/벡터의 길이

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
            angle = np.degrees(angle)  # Convert radian to degree
            angle=np.expand_dims(angle.astype(np.float32),axis=0)#float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다.
            _,results,_,_=knn.findNearest(angle,3)#statue,result,인접값,거리
            #print(results)
            idx=int(results[0][0])
            gesture_name=gesture[idx]

            print(gesture_name)
            cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color=(255, 255, 255), thickness=2)

            #if idx in rps_gesture.keys():
            #    gesture_name = rps_gesture[idx]
            #    cv2.putText(img,text=gesture_name,org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,255,255),thickness=2)



    if not  ret:
        break
    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break