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

form_class = uic.loadUiType("./ui/start.ui")[0]

class UIToolTab(QWidget,form_class):###########################화면구성
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)
        self.logo = QLabel(self)
        pixmap = QPixmap("./images/base2.bmp")
        self.logo.setPixmap(QPixmap(pixmap))
        self.setupUi(self)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setFixedSize(661, 564)
        self.startUIToolTab()

    def startUIToolTab(self):  ####### 페이지 별 동작함수 구현
        self.ToolTab = UIToolTab(self)
        #self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("UIToolTab")
        self.setCentralWidget(self.ToolTab)
        self.ToolTab.StartButton.setStyleSheet(
            '''QPushButton{image:url(./images/startA.png); border:0px;}QPushButton:hover{image:url(./images/startC.png); border:0px;}''')
        self.ToolTab.StopButton.setStyleSheet(
            '''QPushButton{image:url(./images/stopA.png); border:0px;}QPushButton:hover{image:url(./images/stopC.png); border:0px;}''')
        self.ToolTab.MeasureButton.setStyleSheet(
            '''QPushButton{image:url(./images/mearsureA.png); border:0px;}QPushButton:hover{image:url(./images/mearsureC.png); border:0px;}''')
        ########버튼과 같은 ui 연결

        self.ToolTab.StartButton.clicked.connect(self.start_webcam)
        self.ToolTab.StopButton.clicked.connect(self.stop_webcam)
        self.ToolTab.MeasureButton.clicked.connect(self.startM)

        ##########초기화 함수
        self.model = tensorflow.keras.models.load_model('keras_model.h5')
        self.classes = ['Scissors', 'Rock', 'Paper']
        self.capture = cv2.VideoCapture(0)
        self.measureflag = 0

        self.show()

    def startM(self):
        self.measureflag = 1



    def start_webcam(self):

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.ToolTab.lineEdit.setText("start web cam")

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.resize(self.image, (640, 480))
        self.image = cv2.flip(self.image, 1)
        h, w, c = self.image.shape
        self.image = self.image[:, 80:80 + h]
        self.image = cv2.resize(self.image, (224, 224))  # 정사각형으로 자르기
        if self.measureflag:
            img_input = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_input = (img_input.astype(np.float32) / 127.0) - 1.0
            img_input = np.expand_dims(img_input, axis=0)
            prediction = self.model.predict(img_input)
            idx = np.argmax(prediction)
            cv2.putText(self.image, text=self.classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)



        self.displayImage(self.image, 1)

    def stop_webcam(self):
        self.timer.stop()
        self.ToolTab.lineEdit.setText("stop web cam")


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
            self.ToolTab.imglabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())