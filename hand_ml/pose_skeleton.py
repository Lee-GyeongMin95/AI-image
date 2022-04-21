import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))

pose= mp_pose.Pose(
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    #print(results.pose_landmarks.landmark)
    if results.pose_landmarks is not None:

        print(results.pose_landmarks)
        for j,res in enumerate(results.pose_landmarks.landmark):
            #print(res)
            #print(len(res.landmark))
            point = np.zeros((33, 3))

            point[j] = [res.x, res.y, res.z]


            #cv2.putText(img, str(j), (int(lm.x* img.shape[1]), int(lm.y* img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)




        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
             connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))
            #landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255)),
            #connection_drawing_spec =mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(0,0,255))
        )



    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break