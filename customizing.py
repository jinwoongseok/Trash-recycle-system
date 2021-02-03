import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import datetime
exit
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, beepsound, detect_and_beep)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
import numpy
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSlider

from PyQt5.QtCore import Qt

path_dir='C:/log/capture'

flag_ = 0
cnt = 0

class ShowVideo(QtCore.QObject):
    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage) #특정 이벤트가 발생할때 이 시그널 방출

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.brightness = 0     #  brightness, contrast 기본 값 설정
        self.contrast = 10
        self.mode_stat = False
        #self.setGeometry(800,800,800,800)

    # slider
    @QtCore.pyqtSlot(int)
    def update_brightness(self, value):    # bslider의 값이 변경되면 호출될 함수
        self.brightness = value    # slider의 값을 brightness에 저장한다. 이렇게 함으로써 다음 frame은 변경된 brightness 값이 적용되어 출력된다

    @QtCore.pyqtSlot(int)
    def update_contrast(self, value):    # bslider의 값이 변경되면 호출될 함수
        self.contrast = value    # slider의 값을 brightness에 저장한다. 이렇게 함으로써 다음 frame은 변경된 brightness 값이 적용되어 출력된다

    @QtCore.pyqtSlot()
    def startVideo(self, save_img=False): #실질적으로 이 함수에서 바운딩 박스도 찾고 화면 표출도 담당함
        run_video = True
        while run_video:

            out, source, weights, view_img, save_txt, imgsz = \
                opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
            webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith(
                '.txt')
            # Initialize
            set_logging()
            device = select_device(opt.device)
            if os.path.exists(out):
                 shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
            if half:
                model.half()  # to FP16

            # Second-stage classifier
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
                modelc.to(device).eval()

            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = True
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz)
            else:
                view_img = True  # show video
                save_img = True
                dataset = LoadImages(source, img_size=imgsz)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            for path, img, im0s, vid_cap in dataset:
                if type(im0s) == list:  # 웹캠인 경우
                    im0s[0] = cv2.addWeighted(im0s[0], self.contrast / 10, im0s[0], 0, self.brightness)
                else:
                    # img = cv2.addWeighted(img, self.contrast / 10, img, 0, self.brightness)  # 기존 이미지에 변경된 brightness와 contrast를 더한다.
                    if type(im0s) == list:  # 웹캠인 경우
                        im0s[0] = cv2.addWeighted(im0s[0], self.contrast / 10, im0s[0], 0, self.brightness)
                    else:
                        if self.mode_stat:  # img RGB im0s BGR
                            im0s_hsv = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
                            # print("dddddddddd: ", im0s_hsv.shape)
                            im0s_mean = im0s_hsv[..., 2].mean()
                            print(im0s_mean)
                            # print("kkk", img.shape)
                            # print("LKJLKJLKJLKJ:", np.transpose(img).shape)

                            # img_hsv = cv2.cvtColor(np.transpose(img), cv2.COLOR_BGR2HSV)

                            # 100 정도로 맞추기 (255까지 있음)
                            im0s_hsv[..., 2] = np.where(im0s_hsv[..., 2] < 200, im0s_hsv[..., 2] + 40, im0s_hsv[..., 2])
                            # im0s_hsv[..., 2] += 30
                            # im0s_hsv[..., 2] = np.where(im0s_hsv[..., 2] >, , im0s_hsv)
                            # img_hsv[..., 2] += 30

                            im0s = cv2.cvtColor(im0s_hsv, cv2.COLOR_HSV2BGR)
                            # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                            print("트렌스포즈 전", img.shape)
                            # img = np.transpose(img)
                            print("트렌스포즈 후", img.shape)

                            # 밝기, 대조값 범위로 조절하는 방법
                            # if im0s_mean >= 56.:
                            #     self.brightness = 40
                            #     self.contrast = 20
                            # # if im0s_mean >= 57.:
                            # #     self.brightness = 50
                            # #     self.contrast = 20
                            # img = cv2.addWeighted(img, self.contrast / 10, img, 0, self.brightness)  # 기존 이미지에 변경된 brightness와 contrast를 더한다.
                            # im0s = cv2.addWeighted(im0s, self.contrast / 10, im0s, 0, self.brightness)

                            # 이미지 이퀄라이제이션 방법1
                            # im0s_ycrcb = cv2.cvtColor(im0s, cv2.COLOR_BGR2YCrCb)
                            # ycrcb_planes = cv2.split(im0s_ycrcb)  # 밝기 성분(Y)에 대해서만 히스토그램 평활화 수행
                            # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
                            # dst_ycrcb = cv2.merge(ycrcb_planes)
                            # im0s = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

                            # 노이즈를 막기 위해 CLAHE를 사용한 방법2
                            # lab = cv2.cvtColor(im0s, cv2.COLOR_BGR2LAB)
                            # lab_planes = cv2.split(lab)
                            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
                            # lab_planes[0] = clahe.apply(lab_planes[0])
                            # lab = cv2.merge(lab_planes)
                            # im0s = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                            # # im0s = cv2.addWeighted(im0s, 50 / 10, im0s, 0, 90)
                            #
                            # img = np.reshape(img, (640, 640, 3))
                            # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                            # lab_planes = cv2.split(lab)
                            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
                            # lab_planes[0] = clahe.apply(lab_planes[0])
                            # lab = cv2.merge(lab_planes)
                            # img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                            # img = np.reshape(img, (3, 640, 640))

                            # 밝기 전체 픽셀 계산으로
                        else:
                            self.brightness = 10
                            self.contrast = 10

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    else:
                        p, s, im0 = path, '', im0s

                    # im0 = cv2.addWeighted(im0, self.contrast / 10, im0, 0, self.brightness)
                    save_path = str(Path(out) / Path(p).name)
                    txt_path = str(Path(out) / Path(p).stem) + (
                        '_%g' % dataset.frame if dataset.mode == 'video' else '')
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            #print("$$$$$$$$$$$$$",det[0,0],det[0,1],det[0,2],"\n",det[0,3],det[0,4],det[0,5])
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh

                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))

                    if view_img:
                        global cnt
                        global flag_
                        # im0 # 매 프레임 마다 실제 이미지 matrix (윗부분 코드에 있음 im0)
                        #이 부분이 영상크기 리사이즈 해주는 코드
                        # PIL_image = Image.fromarray(im0.astype('uint8'), 'RGB')
                        # resized_image=PIL_image.resize((560,600)) #video size which you want
                        # im0 = numpy.array(resized_image)
                        ##알이씨 원 깜빡임 구현
                        cnt += 1
                        if cnt ==30:
                            if flag_==1:
                                flag_=0
                                cnt = 0
                            else:
                                flag_=1
                                cnt = 0

                        if flag_ == 1:
                            cv2.circle(im0, (15, 20), 10, (0, 0, 255), -1)
                        ##REC draw
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (30, 30)
                        fontScale = 1
                        fontColor = (0, 0, 255)
                        lineType = 2

                        cv2.putText(im0, 'REC',
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (10, 60)
                        fontScale = 0.5
                        fontColor = (255, 255, 255)
                        lineType = 2

                        #현재시간 띄우기
                        now = datetime.datetime.now()
                        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

                        cv2.putText(im0, nowDatetime,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        color_swapped_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  # 이미지 컬러 형식 변환
                        qt_image1 = QtGui.QImage(color_swapped_image.data,
                                                 color_swapped_image.shape[1],  # width
                                                 color_swapped_image.shape[0],  # height
                                                 color_swapped_image.strides[0],
                                                 QtGui.QImage.Format_RGB888)

                        self.VideoSignal1.emit(qt_image1)   # qt_image1이 처리가 다완료된 이미지 그것을 방출
                        loop = QtCore.QEventLoop()
                        QtCore.QTimer.singleShot(1, loop.quit)  # 60 ms, FPS를 담당함 숫자 작을수록 부드러워짐
                        loop.exec_()
                    if save_img:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fourcc = 'mp4v'  # output video codec
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            vid_writer.write(im0)

            if save_txt or save_img:
                print('Results saved to %s' % Path(out))
                if platform.system() == 'Darwin' and not opt.update:  # MacOS
                    os.system('open ' + save_path)
            print('Done. (%.3fs)' % (time.time() - t0))
            break


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.brightness = 0     # 초기 brightness와 contrast 값 설정
        self.contrast = 10
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image) # 영상을 0,0 픽셀부터 그려라 다른 수치 입력시 영상 잘림
        self.image = QtGui.QImage()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.size == 0: # 이미지가 비었으면
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('Project simulation')
        # this will hide the title bar
        # css = """
        # QWidget{
        #    Background: #FFFFFF;
        #    color:black;
        #    font:12px bold;
        #    font-weight:bold;
        #    border-radius: 1px;
        #    height: 11px;
        #}
        #QDialog{
        #    Background-image:url('img/titlebar bg.png');
        #    font-size:12px;
        #    color: black;
        #}
        #QToolButton{
        #    Background:#FFFFFF;
        #    font-size:11px;
        #}
        #QToolButton:hover{
        #    Background: #FFFFFF;
        #    font-size:11px;
        #}
        #"""
        #self.setAutoFillBackground(True)
        #self.setBackgroundRole(QtGui.QPalette.Highlight)
        #self.setStyleSheet(css)


        #self.setStyleSheet("background-image:url(C:/Users/VCLL/Desktop/back__example/1.jpg) ;")
        #self.setStyleSheet("background-color:black;")


        #self.setGeometry(0, 50, 1285, vid_height + 120)
        self.setGeometry(570, 150, 1285, vid_height + 120)
        self.setFixedWidth(900)
        #self.setFixedHeight( vid_height + 120)
        self.setFixedHeight(720)
        self.vid = ShowVideo()
        self.vid.moveToThread(thread)
        self.image_viewer1 = ImageViewer()

        self.vid.VideoSignal1.connect(self.image_viewer1.setImage)

        push_button1 = QtWidgets.QPushButton('Play')
        # push_button1.setMaximumWidth(1068)
        push_button1.setMaximumSize(1068, 40)
        push_button1.setStyleSheet(
                              "QPushButton::hover"
                              "{"
                              "background-color: red;"
                              "border-style: outset;"
                              "border-width: 3px;"
                              "border-radius: 5px;"
                              "border-color: gray;"
                              "font: bold 20px;"
                              "color: black;"
                              "min-width: 10em; padding: 6px;"
                              "}"
            
                              "QPushButton"
                              "{"
                              "background-color: gray;"
                              "border-style: outset;"
                              "border-width: 3px;"
                              "border-radius: 5px;"
                              "border-color: red;"
                              "font: bold 20px;"
                              "color: white;"
                              "min-width: 10em; padding: 6px;"
                              "}"
                              )

        push_button1.clicked.connect(self.vid.startVideo)  # 버튼이 눌린 경우 연결 시작
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_layout.addWidget(self.image_viewer1,0,0,QtCore.Qt.AlignHCenter)
        self.grid_layout.addWidget(push_button1, 1, 0)
        self.grid_layout.setColumnStretch(1, 500)
        self.grid_layout.setColumnStretch(0, 1)
        self.autoBtn()
        # self.trackbar()
        self.log()
        self.grid_layout.setAlignment(Qt.AlignCenter)

        self.layout_widget = QtWidgets.QWidget()
        self.layout_widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.layout_widget)

    def autoBtn(self):
        self.modebtn = QtWidgets.QPushButton('AUTO MODE')

        self.modebtn.setStyleSheet("background-color: beige;"
                                   "border-style: outset; "
                                   "border-width: 5px; "
                                   "border-radius: 35px; "
                                   "border-color: gray; "
                                   "font: bold 17px; "
                                   "color: black;"
                                   "min-width: 10em; padding: 6px;")

        self.modebtn.setFixedSize(130, 70)
        self.modebtn.setCheckable(True)
        self.modebtn.clicked[bool].connect(self.modeBtnClicked)
        self.grid_layout.addWidget(self.modebtn, 1, 1)


    def log(self):
        self.log_layout = QVBoxLayout()
        self.log_layout.setSpacing(3)
        self.log_label = QLabel('LOG 기록')
        self.log_label.setFont(QtGui.QFont('Arial', 28))
        self.log_label.setStyleSheet('color: black')
        #self.log_label.setAlignment(Qt.AlignCenter)

        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setFixedSize(190, 500)
        self.log_layout.addWidget(self.log_label, 0, alignment=Qt.AlignCenter)
        self.log_layout.addWidget(self.tableWidget, 1, alignment=Qt.AlignCenter)

        self.setTableWidgetData()
        self.tableWidget.scrollToBottom()

        self.tableWidget.cellClicked.connect(self.mycell_clicked)
        self.tableWidget.viewport().update()
        # 창 아이콘 설정
        # self.setWindowIcon(QIcon("python.jpg"))
        self.btn = QtWidgets.QPushButton('Update')

        self.btn.setStyleSheet(
                              "QPushButton::hover"
                              "{"
                              "background-color: beige;"
                              "border-style: outset;"
                              "border-width: 5px;"
                              "border-radius: 5px;"
                              "border-color: gray;"
                              "font: bold 17px;"
                              "color: black;"
                              "min-width: 10em; padding: 6px;"
                              "}"

                              "QPushButton"
                              "{"
                              "background-color: gray;"
                              # "border-style: outset; "
                              "border-width: 5px; "
                              "border-radius: 5px; "
                              "border-color: gray; "
                              "font: bold 17px; "
                              "color: white;"
                              "min-width: 10em; padding: 6px;"
                              "}"
                              )

        self.btn.setFixedSize(130, 50)

        self.log_layout.addWidget(self.btn, 2, alignment=Qt.AlignCenter)
        self.btn.clicked.connect(self.btn_clicked)
        self.grid_layout.addLayout(self.log_layout, 0, 1)

    def trackbar(self):
        self.slider_general_layout = QtWidgets.QGridLayout()
        self.slider_dummy_layout = QtWidgets.QGridLayout()
        self.slider_pair_layout = QtWidgets.QGridLayout()

        self.b_slider = QtWidgets.QPushButton()
        self.b_slider.setStyleSheet("background-color: white;")
        self.slider_pair_layout.addWidget(self.b_slider, 0, 1)

        self.slider_pair_layout.setColumnStretch(1, 500)
        self.slider_pair_layout.setColumnStretch(0, 1)

        ##
        self.modebtn = QtWidgets.QPushButton('AUTO MODE')

        self.modebtn.setStyleSheet("background-color: beige;"
                              "border-style: outset; "
                              "border-width: 5px; "
                              "border-radius: 35px; "
                              "border-color: gray; "
                              "font: bold 17px; "
                              "color: black;"
                              "min-width: 10em; padding: 6px;")

        self.modebtn.setFixedSize(130, 70)
        self.modebtn.setCheckable(True)
        self.modebtn.clicked[bool].connect(self.modeBtnClicked)


        self.slider_dummy_layout.addWidget(self.modebtn)

        self.slider_general_layout.addLayout(self.slider_pair_layout, 0, 0)
        self.grid_layout.addLayout(self.slider_general_layout, 3, 0)
        self.grid_layout.addLayout(self.slider_dummy_layout, 3, 1)



    def modeBtnClicked(self, value):
        if value:  # 바뀔 때 (한번 클릭)
            self.modebtn.setStyleSheet("background-color: beige;"
                              "border-style: outset; "
                              "border-width: 5px; "
                              "border-radius: 35px; "
                              "border-color: red; "
                              "font: bold 17px; "
                              "color: black;"
                              "min-width: 10em; padding: 6px;")
            self.vid.mode_stat = True

        else: # 다시 돌아가기 (두번 클릭)
            self.vid.brightness = 0
            self.vid.contrast = 10
            self.modebtn.setStyleSheet("background-color: beige;"
                              "border-style: outset; "
                              "border-width: 5px; "
                              "border-radius: 35px; "
                              "border-color: gray; "
                              "font: bold 17px; "
                              "color: black;"
                              "min-width: 10em; padding: 6px;")
            self.vid.mode_stat = False



    def setTableWidgetData(self):
        self.file_list = os.listdir(path_dir)
        self.tableWidget.setRowCount(len(self.file_list)) # 행의 개수
        self.tableWidget.setColumnCount(1) #열의 개수
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        column_idx_lookup={'file_list' : 0}
        column_headers=['날짜 및 시간']
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        for row, name in enumerate(self.file_list):
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(name[:-4]))
            self.tableWidget.item(row, 0).setBackground(QtGui.QColor(255, 255, 255))

        self.tableWidget.horizontalHeader()
        self.tableWidget.resizeColumnsToContents()

    def btn_clicked(self):
        self.setTableWidgetData()

    def mycell_clicked(self, row, col):
        img_name = path_dir + '/' + self.file_list[row]
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        cv2.imshow('LOG', img)
        key = cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images/', help='source')   # file/folder, 0 for webcam inference/images
    # parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam inference/images
    parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.70, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')

    global opt
    global vid_height
    vid_height = 800

    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                app = QtWidgets.QApplication(sys.argv)
                thread = QtCore.QThread()
                thread.start()
                window = MyWindow()
                window.repaint()
                window.show()
                app.exec_()
        else:
            app = QtWidgets.QApplication(sys.argv)
            thread = QtCore.QThread()
            thread.start()
            window=MyWindow()
            window.repaint()
            window.show()
            app.exec_()