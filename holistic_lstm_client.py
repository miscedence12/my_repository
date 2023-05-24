#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from lstm_hand.lstm import WINDOW_SIZE
from lstm_hand.lstm import ActionClassificationLSTM
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageFont
import socket

lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("lightning_logs/version_5\checkpoints\epoch=9-step=689.ckpt")
lstm_classifier.eval()
# 关键点索引
KEYPOINT_DICT = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_finger_mcp': 5,
    'index_finger_pip': 6,
    'index_finger_dip': 7,
    'index_finger_tip': 8,
    'middle_finger_mcp': 9,
    'middle_finger_pip': 10,
    'middle_finger_dip': 11,
    'middle_finger_tip': 12,
    'ring_finger_mcp': 13,
    'ring_finger_pip': 14,
    'ring_finger_dip': 15,
    'ring_finger_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}
###定义手势标签
LABELS = {
    0: "机器人停止",
    1: "机器人右转°",
    2: "机器人左转",
    3: "机器人后退",
    4: "机器人前进",
}
l="未识别到动作"
mp_drawing = mp.solutions.drawing_utils

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument('--unuse_smooth_landmarks', action='store_true')
    parser.add_argument('--enable_segmentation', action='store_true')
    parser.add_argument('--smooth_segmentation', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--segmentation_score_th",
                        help='segmentation_score_threshold',
                        type=float,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main():
    global l
    # 参数解析 #################################################################
    args = get_args()
    cap_device = args.device  # 0
    cap_width = args.width  # 960
    cap_height = args.height  # 540

    # upper_body_only = args.upper_body_only
    smooth_landmarks = not args.unuse_smooth_landmarks  # True
    enable_segmentation = args.enable_segmentation  # False
    smooth_segmentation = args.smooth_segmentation  # False
    model_complexity = args.model_complexity  # 1
    min_detection_confidence = args.min_detection_confidence  # 0.5
    min_tracking_confidence = args.min_tracking_confidence  # 0.5
    segmentation_score_th = args.segmentation_score_th  # 0.5

    use_brect = args.use_brect  # False
    plot_world_landmark = args.plot_world_landmark  # False

    # 设置视频流 ###############################################################
    # cap = cv.VideoCapture("rtsp://admin:okwy1234@192.168.100.64:554/h264/ch1/main/av_stream")
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 定义检测函数 #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        smooth_segmentation=smooth_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS计算 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    # World坐标系 ####################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
    ###########
    buffer_window = []
    ############


    data_socket=Client_connect()
    biaozhiwei=data_socket.recv(1024).decode("utf8")
    print("接收到来自服务器端的标志位：",biaozhiwei)


    if biaozhiwei=="1":
        while True:
            # print("标志位：",biaozhiwei)
            b=[]
            display_fps = cvFpsCalc.get()
            # 相机捕捉 #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # 水平翻转
            debug_image = copy.deepcopy(image)

            #检测#############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            # print(results)
            image.flags.writeable = True

            # Pose ###############################################################
            if enable_segmentation and results.segmentation_mask is not None:  # False
                mask = np.stack((results.segmentation_mask,) * 3,
                                axis=-1) > segmentation_score_th
                bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
                bg_resize_image[:] = (0, 255, 0)
                debug_image = np.where(mask, debug_image, bg_resize_image)

            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                # 外接矩形
                # brect = calc_bounding_rect(debug_image, pose_landmarks)  # [267,409,601,540]
                # 描画
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                for index, landmark in enumerate(pose_landmarks.landmark):
                    if index==12 or index==14 or index==16:
                        pose_x = min(int(landmark.x * cap_width), cap_width - 1)
                        pose_y = min(int(landmark.y * cap_height), cap_height - 1)
                        if pose_x>0 and pose_y>0:
                            b.append(str(pose_x))
                            b.append(str(pose_y) )

                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            # Pose:World坐标系 #############################################
            if plot_world_landmark:  # False
                if results.pose_world_landmarks is not None:
                    plot_world_landmarks(
                        plt,
                        ax,
                        results.pose_world_landmarks,
                    )

            # Hands ########################################################
            # left_hand_landmarks = results.left_hand_landmarks  # None
            right_hand_landmarks = results.right_hand_landmarks
            # # 左手
            # if left_hand_landmarks is not None:
            #     # 手掌重心計算
            #     cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
            #     # 外接矩形的計算
            #     brect = calc_bounding_rect(debug_image, left_hand_landmarks)
            #     # 描画
            #     debug_image = draw_hands_landmarks(
            #         debug_image,
            #         cx,
            #         cy,
            #         left_hand_landmarks,
            #         # upper_body_only,
            #         'R',
            #     )
            #     debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 右手
            if right_hand_landmarks is not None:
                # 手的平重心計算
                # cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 外接矩形的計算
                # brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # 描画
                # debug_image = draw_hands_landmarks(
                #     debug_image,
                #     cx,
                #     cy,
                #     right_hand_landmarks,
                #     # upper_body_only,
                #     'L',
                # )

                for index, landmark in enumerate(right_hand_landmarks.landmark):
                    if landmark.visibility < 0 or landmark.presence < 0:
                            continue
                    palm_x = min(int(landmark.x * cap_width), cap_width - 1)
                    palm_y = min(int(landmark.y * cap_height), cap_height - 1)
                    if palm_x>0 and palm_y>0:
                        b.append(str(palm_x))
                        b.append(str(palm_y))
                # print("b=",b)
                # print("len(b)",len(b))

                # 关键点可视化
                mp_drawing.draw_landmarks(
                    debug_image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            features = []
            if len(b)==48:
                features.append(b)
                if len(buffer_window) < WINDOW_SIZE:  # WINDOW_SIZE = 32
                    buffer_window.append(features)
                else:
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    # model_input = torch.unsqueeze(model_input, dim=0)  # tensor(1,32,34)
                    model_input=model_input.permute(1,0,2)
                    y_pred = lstm_classifier(model_input)
                    # print(y_pred)
                    prob = F.softmax(y_pred, dim=1)  ####softmax操作
                    # print(prob)
                    pred_index = prob.data.max(dim=1)[1]  ##tensor([i]) i为动作的索引值
                    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]
                    l = label
                    # print("Label detected ", l)
            else:
                l="未识别到动作"

            # FPS
            if enable_segmentation and results.segmentation_mask is not None:
                fps_color = (255, 255, 255)
            else:
                fps_color = (0, 255, 0)
            debug_image=cv2AddChineseText(debug_image, "动作:" + l,position= (10, 40),textColor=(255, 0, 0),textSize= 30)
            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2, cv.LINE_AA)
            

            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映 #############################################################
            cv.imshow('MediaPipe Holistic Demo', debug_image)
            Send_data(data_socket,1024,l)
    else:
        print("标志位：",biaozhiwei)
        print("标志位不为1,无法启动手势交互功能")
    cap.release()
    cv.destroyAllWindows()

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        # if index == 0:  # 手首1
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 1:  # 手首2
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 5:  # 人差指
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 9:  # 中指
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 13:  # 薬指
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 17:  # 小指
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


# def draw_hands_landmarks(
#         image,
#         cx,
#         cy,
#         landmarks,
#         # upper_body_only,
#         handedness_str='R'):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     landmark_point = []
#
#     # 遍历
#     for index, landmark in enumerate(landmarks.landmark):
#         if landmark.visibility < 0 or landmark.presence < 0:
#             continue
#
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         landmark_z = landmark.z
#
#         landmark_point.append((landmark_x, landmark_y))
#
#         if index == 0:  # 手首1
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 1:  # 手首2
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 2:  # 親指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 3:  # 親指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 4:  # 親
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 5:  # 人差指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 6:  # 人差指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 7:  # 人差指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 8:  # 人差指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 9:  # 中指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 10:  # 中指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 11:  # 中指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 13:  # 薬指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 14:  # 薬指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 15:  # 薬指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 16:  # 薬指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 17:  # 小指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 18:  # 小指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 19:  # 小指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 20:  # 小指
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#
#         # # if not upper_body_only:
#         # if True:
#         #     cv.putText(image, "z:" + str(round(landmark_z, 3)),
#         #                (landmark_x - 10, landmark_y - 10),
#         #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
#         #                cv.LINE_AA)
#
#     # 连接线
#     if len(landmark_point) > 0:
#         # 拇指
#         cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
#         cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)
#
#         # 人差指
#         cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
#         cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
#         cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)
#
#         # 中指
#         cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
#         cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
#         cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)
#
#         # 无名指
#         cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
#         cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
#         cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)
#
#         # 小指
#         cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
#         cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
#         cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)
#
#         # 棕榈
#         cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
#         cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
#         cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
#         cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
#         cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
#         cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
#         cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)
#
#     # 重心 + 左右
#     if len(landmark_point) > 0:
#         cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
#         cv.putText(image, handedness_str, (cx - 6, cy + 6),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
#
#     return image


def draw_pose_landmarks(
        image,
        landmarks,
        # upper_body_only,
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        # if index == 0:  # 鼻
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 1:  # 右目：目頭
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 2:  # 右目：瞳
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 3:  # 右目：目尻
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 4:  # 左目：目頭
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 5:  # 左目：瞳
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 6:  # 左目：目尻
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 7:  # 右耳
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 8:  # 左耳
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 9:  # 口：左端
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 10:  # 口：左端
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 11:  # 右肩
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 0, 255), 2)
        # if index == 13:  # 右肘
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 0, 255), 2)
        # if index == 15:  # 右手首
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 0, 255), 2)
        # if index == 17:  # 右手1(外側端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 18:  # 左手1(外側端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 19:  # 右手2(先端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 20:  # 左手2(先端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 21:  # 右手3(内側端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 22:  # 左手3(内側端)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 23:  # 腰(右側)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 24:  # 腰(左側)
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 25:  # 右ひざ
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 26:  # 左ひざ
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 27:  # 右足首
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 28:  # 左足首
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 29:  # 右かかと
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 30:  # 左かかと
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 31:  # 右つま先
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        # if index == 32:  # 左つま先
        #     cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # # if not upper_body_only:
        # if True:
        #     cv.putText(image, "z:" + str(round(landmark_z, 3)),
        #                (landmark_x - 10, landmark_y - 10),
        #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        #                cv.LINE_AA)

    if len(landmark_point) > 0:
        # # 右目
        # if landmark_point[1][0] > visibility_th and landmark_point[2][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[1][1], landmark_point[2][1],
        #             (0, 255, 0), 2)
        # if landmark_point[2][0] > visibility_th and landmark_point[3][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[2][1], landmark_point[3][1],
        #             (0, 255, 0), 2)

        # # 左目
        # if landmark_point[4][0] > visibility_th and landmark_point[5][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[4][1], landmark_point[5][1],
        #             (0, 255, 0), 2)
        # if landmark_point[5][0] > visibility_th and landmark_point[6][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[5][1], landmark_point[6][1],
        #             (0, 255, 0), 2)

        # # 口
        # if landmark_point[9][0] > visibility_th and landmark_point[10][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[9][1], landmark_point[10][1],
        #             (0, 255, 0), 2)

        # 肩
        # if landmark_point[11][0] > visibility_th and landmark_point[12][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[11][1], landmark_point[12][1],
        #             (0, 255, 0), 2)

        # 右腕
        # if landmark_point[11][0] > visibility_th and landmark_point[13][
        #     0] > visibility_th:
        #     cv.line(image, landmark_point[11][1], landmark_point[13][1],
        #             (0, 255, 0), 2)
        # if landmark_point[13][0] > visibility_th and landmark_point[15][
        #     0] > visibility_th:
        #     cv.line(image, landmark_point[13][1], landmark_point[15][1],
        #             (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        # if landmark_point[15][0] > visibility_th and landmark_point[17][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[15][1], landmark_point[17][1],
        #             (0, 255, 0), 2)
        # if landmark_point[17][0] > visibility_th and landmark_point[19][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[17][1], landmark_point[19][1],
        #             (0, 255, 0), 2)
        # if landmark_point[19][0] > visibility_th and landmark_point[21][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[19][1], landmark_point[21][1],
        #             (0, 255, 0), 2)
        # if landmark_point[21][0] > visibility_th and landmark_point[15][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[21][1], landmark_point[15][1],
        #             (0, 255, 0), 2)
        #
        # # 左手
        # if landmark_point[16][0] > visibility_th and landmark_point[18][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[16][1], landmark_point[18][1],
        #             (0, 255, 0), 2)
        # if landmark_point[18][0] > visibility_th and landmark_point[20][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[18][1], landmark_point[20][1],
        #             (0, 255, 0), 2)
        # if landmark_point[20][0] > visibility_th and landmark_point[22][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[20][1], landmark_point[22][1],
        #             (0, 255, 0), 2)
        # if landmark_point[22][0] > visibility_th and landmark_point[16][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[22][1], landmark_point[16][1],
        #             (0, 255, 0), 2)

        # 胴体
        # if landmark_point[11][0] > visibility_th and landmark_point[23][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[11][1], landmark_point[23][1],
        #             (0, 255, 0), 2)
        # if landmark_point[12][0] > visibility_th and landmark_point[24][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[12][1], landmark_point[24][1],
        #             (0, 255, 0), 2)
        # if landmark_point[23][0] > visibility_th and landmark_point[24][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[23][1], landmark_point[24][1],
        #             (0, 255, 0), 2)

        # if len(landmark_point) > 25:
        #     # 右足
        #     if landmark_point[23][0] > visibility_th and landmark_point[25][
        #             0] > visibility_th:
        #         cv.line(image, landmark_point[23][1], landmark_point[25][1],
        #                 (0, 255, 0), 2)
        #     if landmark_point[25][0] > visibility_th and landmark_point[27][
        #             0] > visibility_th:
        #         cv.line(image, landmark_point[25][1], landmark_point[27][1],
        #                 (0, 255, 0), 2)
        #     if landmark_point[27][0] > visibility_th and landmark_point[29][
        #             0] > visibility_th:
        #         cv.line(image, landmark_point[27][1], landmark_point[29][1],
        #                 (0, 255, 0), 2)
        #     if landmark_point[29][0] > visibility_th and landmark_point[31][
        #             0] > visibility_th:
        #         cv.line(image, landmark_point[29][1], landmark_point[31][1],
        #                 (0, 255, 0), 2)

        # # 左足
        # if landmark_point[24][0] > visibility_th and landmark_point[26][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[24][1], landmark_point[26][1],
        #             (0, 255, 0), 2)
        # if landmark_point[26][0] > visibility_th and landmark_point[28][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[26][1], landmark_point[28][1],
        #             (0, 255, 0), 2)
        # if landmark_point[28][0] > visibility_th and landmark_point[30][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[28][1], landmark_point[30][1],
        #             (0, 255, 0), 2)
        # if landmark_point[30][0] > visibility_th and landmark_point[32][
        #         0] > visibility_th:
        #     cv.line(image, landmark_point[30][1], landmark_point[32][1],
        #             (0, 255, 0), 2)
    return image


def plot_world_landmarks(
        plt,
        ax,
        landmarks,
        visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 脸
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

    plt.pause(.001)

    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image
def Client_udp(send_data):

    # 创建udp套接字,
    # AF_INET表示ip地址的类型是ipv4，
    # SOCK_DGRAM表示传输的协议类型是udp
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
    # 要发送的ip地址和端口（元组的形式）
    send_addr = ('192.168.171.10', 10085)
    udp_socket.connect(send_addr)
    # print ('send_addr = ', send_addr)
 
    # while True:
    # 要发送的信息
    test_data = send_data
    # print ('send_data: ', test_data)

    # 发送消息
    # udp_socket.sendto(test_data, send_addr)
    udp_socket.send(test_data.encode("utf8"))
    # data=udp_socket.recvfrom(1024)
    # print(data.decode("utf8"))

    # 关闭套接字
    # udp_socket.close()
def Client_connect():
    #  === TCP 客户端程序 client.py ===
    IP = '192.168.171.10'
    SERVER_PORT = 10086
    BUFLEN = 1024

    # 实例化一个socket对象，指明TCP协议
    dataSocket = socket.socket(socket.AF_INET,socket. SOCK_STREAM)

    # 连接服务端socket
    dataSocket.connect((IP, SERVER_PORT))
    print("TCP连接建立成功......")
    return dataSocket

def Send_data(dataSocket,BUFLEN,send_data):
    # 从终端读入用户输入的字符串
    toSend = send_data
    # 发送消息，也要编码为 bytes
    dataSocket.send(toSend.encode("utf8"))

    # 等待接收服务端的消息
    #非阻塞模式
    dataSocket.setblocking(0)
    try:
        recved = dataSocket.recv(BUFLEN)
        # 打印读取的信息
        print(recved.decode())
    except BlockingIOError as e:
        #阻塞模式
        dataSocket.setblocking(1)
    # 如果返回空bytes，表示对方关闭了连接
 
    

    # dataSocket.close()

if __name__ == '__main__':
    main()
