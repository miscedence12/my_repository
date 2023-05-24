#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
mp_drawing = mp.solutions.drawing_utils
f=open("./Database3/X_train.txt", "a", encoding="utf-8")
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default))',
                        type=int,
                        default=1)

    parser.add_argument("--max_num_hands", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.8)

    parser.add_argument('--use_brect' ,action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 参数解析 #################################################################

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # 设置摄像头编号和采样视频分辨率
    cap = cv.VideoCapture("output/database3/train/a1.avi")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=model_complexity,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS测量模块 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 世界坐标图 ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        r_ax = fig.add_subplot(121, projection="3d")
        l_ax = fig.add_subplot(122, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:
        a = []
        display_fps = cvFpsCalc.get()

        # 相机捕捉 #####################################################
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        #因为摄像头是镜像的，所以将摄像头水平翻转,不是镜像的可以不翻转
        debug_image = copy.deepcopy(image)

        # 进行检测 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        #----------------------------------------------------------------------------#
        # if results.multi_handedness:
        #     for hand_label in results.multi_handedness:
        #         print(hand_label)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id ,lm in enumerate(hand_landmarks.landmark):
                # print('hand_landmarks:',hand_landmarks)
                    cx,cy=int(lm.x*image.shape[1]),int(lm.y*image.shape[0])  #x_w.y_h
                    x_w,y_h=str(cx),str(cy)
                    a.append(x_w)
                    a.append(",")
                    a.append(y_h+",")
                    # print(id, cx, cy)
                a[-1]=a[-1].split(",")[0]+"\n"
                print(a)
                print("len(a)",len(a))
                f.writelines(a)

                # 关键点可视化
                mp_drawing.draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('MediaPipe Hands', debug_image)
        if cv.waitKey(1) & 0xFF == 27:
            break
    cap.release()
if __name__ == '__main__':
    main()
    f.close()
