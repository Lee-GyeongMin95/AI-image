from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import sys

import cv2

import tensorflow.keras
import numpy as np

form_class = uic.loadUiType("./ui/camimg.ui")[0] # ui 불러오기

class UIToolTab(QWidget,form_class):###########################화면구성/한 화면당 1개의 클래스 생성
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)
        self.logo = QLabel(self) ## 배경화면구성

        pixmap = QPixmap("./images/tiger08.jpg")## 배경화면구성
        pixmap = pixmap.scaled(687, 729)
        self.logo.setPixmap(QPixmap(pixmap))## 배경화면구성
        self.setupUi(self)

class MainWindow(QMainWindow):##메인 프로그램
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setFixedSize(687, 729)## 프래그램의 전체 크기
        self.startUIToolTab()

    def startUIToolTab(self):  ####### 페이지 별 동작함수 구현
        self.ToolTab = UIToolTab(self)
        #self.setWindowFlag(Qt.FramelessWindowHint) ## 상태창이 없다. 프레임이 없다.
        self.setWindowTitle("UIToolTab")
        self.setCentralWidget(self.ToolTab)

        self.ToolTab.startButton.setStyleSheet(
            '''QPushButton{image:url(./images/startA.png); border:0px;}QPushButton:hover{image:url(./images/startC.png); border:0px;}''')
        self.ToolTab.stopButton.setStyleSheet(
            '''QPushButton{image:url(./images/stopA.png); border:0px;}QPushButton:hover{image:url(./images/stopC.png); border:0px;}''')
        self.ToolTab.MeasureButton.setStyleSheet(
            '''QPushButton{image:url(./images/mearsureA.png); border:0px;}QPushButton:hover{image:url(./images/mearsureC.png); border:0px;}''')
        self.ToolTab.saveButton.setStyleSheet(
            '''QPushButton{image:url(./images/save1.jpg); border:0px;}QPushButton:hover{image:url(./images/save2.jpg); border:0px;}''')

        # ########버튼과 같은 ui 연결
        self.ToolTab.startButton.clicked.connect(self.start_webcam)
        self.ToolTab.stopButton.clicked.connect(self.stop_webcam)
        self.ToolTab.MeasureButton.clicked.connect(self.startM)
        self.ToolTab.saveButton.clicked.connect(self.save_webcam)
        ##########프로그램 set up
        self.model = tensorflow.keras.models.load_model('keras_model.h5')
        self.classes = ['Scissors', 'Rock', 'Paper']
        self.capture = cv2.VideoCapture(0)
        self.measureflag = 0

        self.show()

    def startM(self):
        self.measureflag = 1

    def save_webcam(self):
        cv2.imwrite('images/saveimage/saveimg.jpg',self.image) # image 저장

    def stop_webcam(self):
        self.timer.stop()
        self.ToolTab.textBrowser.setText("stop web cam")
        self.measureflag = 0

    def start_webcam(self):
        # self.ToolTab.logo.hide()
        self.timer = QTimer(self)## timer interrupt
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.ToolTab.textBrowser.setText("start web cam")
    def update_frame(self): ## 영상처리 하는 부분
        ret, self.image = self.capture.read() ## capture.read 의 반환값은 2개이다.
        self.displayImage(self.image)#창에 바로 뛰우기, 화면

        self.image = cv2.resize(self.image, (640, 480))
        self.image = cv2.flip(self.image, 1)

        h, w, c = self.image.shape
        self.image = self.image[:, 80:80 + h]##crop image
        self.image = cv2.resize(self.image, (224, 224))
        ## teachable machine model image 가 224,224 여서 resize
        if self.measureflag:
            img_input = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_input = (img_input.astype(np.float32) / 127.0) - 1.0
            img_input = np.expand_dims(img_input, axis=0)
            prediction = self.model.predict(img_input)
            idx = np.argmax(prediction)
            cv2.putText(self.image, text=self.classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2)

        self.displayImage(self.image, 1)



    # def update_frame(self):
    #     ret, self.image = self.capture.read()
    #     self.image = cv2.resize(self.image, (640, 480))
    #     self.image = cv2.flip(self.image, 1)
    #     h, w, c = self.image.shape
    #     self.image = self.image[:, 80:80 + h]
    #     self.image = cv2.resize(self.image, (224, 224))  # 정사각형으로 자르기
    #     if self.measureflag:
    #         img_input = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    #         img_input = (img_input.astype(np.float32) / 127.0) - 1.0
    #         img_input = np.expand_dims(img_input, axis=0)
    #         prediction = self.model.predict(img_input)
    #         idx = np.argmax(prediction)
    #         cv2.putText(self.image, text=self.classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    #                     color=(255, 255, 255), thickness=2)
    #
    #
    #
    #     self.displayImage(self.image, 1)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.ToolTab.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.ToolTab.imglabel.setScaledContents(T


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())