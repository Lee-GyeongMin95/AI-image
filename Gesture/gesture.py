import tensorflow.keras
import numpy as np
import cv2

model=tensorflow.keras.models.load_model('keras_model.h5')
cap =cv2.VideoCapture(0)

classes=['Scissors','Rock','Paper']

while cap.isOpened():
    ret,img=cap.read()
    img=cv2.flip(img,1)
    h,w,c=img.shape

    img=img[:,80:80+h]
    print(img.shape)
    img=cv2.resize(img,(224,224))# 정사각형으로 자르기

    img_input =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow('BGR View', img_input)

    img_input=(img_input.astype(np.float32)/127.0)-1.0
    img_input=np.expand_dims(img_input,axis=0)
    print(img_input.shape)
    prediction =model.predict(img_input)
    print(prediction)
    idx=np.argmax(prediction)
    cv2.putText(img,text=classes[idx],org=(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)


    if not  ret:
        break

    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break
