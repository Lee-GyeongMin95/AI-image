import tensorflow.keras
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('keras_model.h5')
cap = cv2.VideoCapture(0)

classes=['Scissors','Rock','Paper']

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape  # webcam image size check
    # print(h,w,c)
    img = img[:,80:80+h]#crop image
    # print(img.shape)
    img=cv2.resize(img,(224,224))
    img_input = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#opencv image(bgr) -> model(rgb)
    # cv2.imshow('BGR view', img_input)
    img_input = (img_input.astype(np.float32)/127)-1.0
    #color map 0~255, keras color map -1~1 이여서 변환 필요. astype 쓰는 이유는 원래 값은 int 값이여서
    img_input = np.expand_dims(img_input, axis=0)
    #keras model, tensorflow model 은 축이 하나 더 있다. 그래서 축 추가 필요 ,axis=0 이면 맨앞, -1 이면 맨 뒤에 생성
    # print(img_input)
    result = model.predict(img_input)
    # print('result',result)
    idx = np.argmax(result) # 제일 큰 index 의 값을 반환한다.
    cv2.putText(img, ## img 에
                text=classes[idx], # text 를 적을꺼다
                org=(10,30), # 10,30 위치에 시작해서
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,#폰트
                fontScale=1,color=(255,255,255),#글자 색깔
                thickness=2) #글자의 두께

    if not ret:
        break
    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break