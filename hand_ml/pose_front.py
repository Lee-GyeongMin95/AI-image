from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np

stop=0
cap = cv2.VideoCapture('./samplevideo/Overhead Squat(side).mp4')
#cap = cv2.VideoCapture(0)
detector = PoseDetector()
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=200)
#peachBtP=np.zeros((1, 3))
while True:
    success, img = cap.read()
    #copy_img = img.copy()

    copy_img = fgbg.apply(img)
    h, w, c = img.shape

    img ,d= detector.findPose(img)
    if d.pose_landmarks is not None:

        #joint = np.zeros((33, 3))
        joint=[]
        peachBtP=[]
        earBtP = []
        shoulderBtP=[]
        waistBtP=[]
        for j, lm in enumerate(d.pose_landmarks.landmark):  # 21개의 랜드 마크가 들어있는데 한점씩 반복문을 사용하여 처리한다.
            joint.append([int(lm.x*w), int(lm.y*h), int(lm.z*w)])
            #landmarks.append((int(lm.x* width), int(lm.y * height),(lm.z * width)))
            #print(jointp[])
        peachBtP=[int((joint[28][0]+joint[27][0])/2),int((joint[28][1]+joint[27][1])/2),int((joint[28][2]+joint[27][2])/2)]
        earBtP = [int((joint[7][0] + joint[8][0]) / 2), int((joint[7][1] + joint[8][1]) / 2),
                    int((joint[7][2] + joint[8][2]) / 2)]
        shoulderBtP = [int((joint[11][0] + joint[12][0]) / 2), int((joint[11][1] + joint[12][1]) / 2),
                  int((joint[11][2] + joint[12][2]) / 2)]
        waistBtP = [int((joint[23][0] + joint[24][0]) / 2), int((joint[23][1] + joint[24][1]) / 2),
                  int((joint[23][2] + joint[24][2]) / 2)]
        #peachBtP[1] = joint[28][1] + joint[27][1]
        #peachBtP[2] = joint[28][2] + joint[27][2]
        #angle2 = detector.findAngle_p(img, joint[0] , peachBtP, joint[27], draw=True)
        angle2 = detector.findAngle_p(img, earBtP , peachBtP, joint[27], (255,255,0), (255,255,0), (255,255,0), draw=True)### line,circle,text
        angle3 = detector.findAngle_p(img, earBtP, shoulderBtP, joint[11], (255, 0, 255), (255, 0, 255), (255, 0, 255),
                                      draw=True)
        angle4 = detector.findAngle_p(img, earBtP, waistBtP, joint[23], (0, 255, 255), (0, 255, 255), (0, 255, 255),
                                      draw=True)
        print(peachBtP)

    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
#    print(type(lmList[0][1:]))
    angle0=detector.findAngle(img, 23, 25, 27, draw=True)
    angle1 = detector.findAngle(img, 24, 26, 28, draw=True)
    #angle2 = detector.findAngle_p(img, lmList[23][1:], lmList[25][1:], lmList[27][1:], draw=True)

#    imgray = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)  # 이미지 그레이 전환

   # ret, thresh = cv2.threshold(copy_img, 200, 255,0)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # for contour in contours:
    #    copy_img=cv2.drawContours(copy_img,[contour],-1,(0,255,0),2)



    cv2.imshow("Image",copy_img)



    if cv2.waitKey(1) & 0xFF == ord('s'):
        stop=1
        while(stop==1):
            if cv2.waitKey(1) & 0xFF == ord('d'): stop=0

    #print(bboxInfo)
    if bboxInfo:
        center = bboxInfo["center"]
        cv2.circle(img, center, 15, (255, 0, 255), 5)

    cv2.imshow("Image2", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()