#### gui 쓸때 그냥 다 쓰자
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
#######
import cv2
import numpy as np
import sys
import mediapipe as mp
import csv

import utils.HandTrackingModule as htm
form_class = uic.loadUiType("./ui/mygesturegui.ui")[0]

class UIToolTab(QWidget,form_class):  ###########################화면구성
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)
        self.setupUi(self)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setFixedSize(664, 560)
        self.startUIToolTab()

    def startUIToolTab(self):  ####### 페이지 별 동작함수 구현
        self.ToolTab = UIToolTab(self)
        # self.setWindowFlag(Qt.FramelessWindowHint)  ## 프레임 창
        self.setWindowTitle("UIToolTab")
        self.setCentralWidget(self.ToolTab)
        self.show()

        self.ToolTab.startButton.clicked.connect(self.start_function)
        self.ToolTab.stopButton.clicked.connect(self.stop_function)
        ####초기화 작업
        self.flag=0
        self.count=0
        self.capture = cv2.VideoCapture(0)
        self.detector = htm.HandDetector(maxHands=1, detectionCon=0.75)

        self.start_webcam()

    def start_function(self):
        self.flag=1

    def stop_function(self):
        if self.flag==1:
            self.count =self.count+1
        self.flag = 0

    def start_webcam(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def update_frame(self):##while
        self.name = self.ToolTab.filename.text()
        cam_ret, self.cam_img = self.capture.read()
        self.cam_img = cv2.flip(self.cam_img, 1)
        self.hands, self.cam_img = self.detector.findHands(self.cam_img, flipType=True)

        if self.hands:
            lm_list = self.hands[0]['lmList']
            lm_list = np.array(lm_list)  ## 벡터 연산 쉽게 할라고 배열로 만든다.
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
            angle=np.append(angle,np.array(self.count)) ## count는 제스쳐의 갯수를 세는거임.
            # angle = np.expand_dims(angle.astype(np.float32), axis=0)
            # float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다.
            if self.flag == 1:
                str = self.name + '.csv'
                f = open(str, 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(angle)
                f.close()

        self.displayImage(self.cam_img, 1)

    def stop_webcam(self):
        self.timer.stop()
        # self.capture.release()

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