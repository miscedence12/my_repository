#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ###################################
# @author:Fxj
# @email:fxjswjtu@my.swjtu.edu.cn
# ###################################
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from lstm_hand.lstm import WINDOW_SIZE
from lstm_hand.lstm import ActionClassificationLSTM,ActionClassificationLSTM2
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageFont
import signal
import logging
import time
import depthai as dai
import huidiao.gesture

pipeline=dai.Pipeline()
# logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

#单臂动作分类模型
lstm_classifier1 = ActionClassificationLSTM.load_from_checkpoint("lightning_logs/version_10\checkpoints\epoch=70-step=24210.ckpt")
if lstm_classifier1:
    logging.info("单臂动作模型加载成功")
lstm_classifier1.eval()

#定义源和输出
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

gesture =huidiao.gesture.Gesture('192.168.31.208', 8080)

# 属性
camRgb.setPreviewSize(960, 540)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
# 连接
camRgb.preview.link(xoutRgb.input)

#双臂动作分类模型
# lstm_classifier2 = ActionClassificationLSTM2.load_from_checkpoint("")
# if lstm_classifier2:
#     logging.info("双臂动作模型加载成功")
# lstm_classifier2.eval()

# Dictionary that maps from joint names to keypoint indices.
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
###定义单臂手势标签手势标签
LABELS1 = {
    0: "机器人停止",
    1: "机器人右转",
    2: "机器人左转",
    3: "机器人后退",
    4: "机器人前进",
}
#定义双臂手势标签
LABELS2={

}
#全局变量
l="未识别到动作"
index=0
biaozhiwei=0
data_json={"robot":1,"start":0}
open_gesture=0
isExit = False
##############
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


def main(data):
    global l
    global index
    global biaozhiwei
    global data_json
    global open_gesture
    global isExit
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
    # cap = cv.VideoCapture("rtsp://admin:okwy1234@192.168.100.64:554/h264/ch1/sub/av_stream")
    # cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

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
    # World坐标系 ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
    ###########
    buffer_window1 = []    #单臂
    buffer_window2 = []    #双臂
    #######另一个线程接收开启关闭标志#####
    Task()
    ####################################
    print("---------识别和关闭识别部分---------------")
    ############
    while not isExit:
        time.sleep(1)
        # data_dic=data_json
        if open_gesture==1:
            data_ack={"robot":1,"ack":1} ##1表示反馈成功，0表示反馈失败
            gesture.sendMsg(3150, data_ack)
            # Send_data(data_socket,1024,json.dumps(data_ack,ensure_ascii=False,default=default_dump))
            with dai.Device(pipeline) as device:
                print('Connected cameras: ', device.getConnectedCameras())
                # 输出USB传输速度
                print('Usb speed: ', device.getUsbSpeed().name)
                # 输出序列可以得到RGB图像
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                while not isExit:
                    b=[]#左手关键点
                    a=[]#右手关键点
                    display_fps = cvFpsCalc.get()
                    inRgb = qRgb.get()
                    image = inRgb.getCvFrame()

                 
                     # 水平翻转
                    if image is None:
                        data_ack={"robot":1,"ack":0} ##1表示反馈成功，0表示反馈失败
                        gesture.sendMsg(3150, data_ack)
                    image = cv.flip(image, 1)
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
                            #采集左臂三个点
                            if index==12 or index==14 or index==16:
                                pose_xl = min(int(landmark.x * cap_width), cap_width - 1)
                                pose_yl = min(int(landmark.y * cap_height), cap_height - 1)
                                if pose_xl>0 and pose_yl>0:
                                    b.append(str(pose_xl))
                                    b.append(str(pose_yl) )

                            #采集右臂三个点
                            if index==11 or index==13 or index==15:
                                pose_xr = min(int(landmark.x * cap_width), cap_width - 1)
                                pose_yr = min(int(landmark.y * cap_height), cap_height - 1)
                                if pose_xr>0 and pose_yr>0:
                                    a.append(str(pose_xr))
                                    a.append(str(pose_yr))

                        # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

                    # Pose:World坐标系 #############################################
                    if plot_world_landmark:  # False
                        if results.pose_world_landmarks is not None:
                            plot_world_landmarks(
                                plt,
                                ax,
                                results.pose_world_landmarks,
                            )

                    #Hands ###############################################################
                    left_hand_landmarks = results.left_hand_landmarks  # None
                    right_hand_landmarks = results.right_hand_landmarks
                    # # 右手
                    if left_hand_landmarks is not None:
                        #手掌重心計算
                        # cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                        #外接矩形的計算
                        # brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                        #找到手的位置点
                        for index, landmark in enumerate(left_hand_landmarks.landmark):
                            if landmark.visibility < 0 or landmark.presence < 0:
                                    continue
                            palm_xr = min(int(landmark.x * cap_width), cap_width - 1)
                            palm_yr = min(int(landmark.y * cap_height), cap_height - 1)
                            if palm_xr>0 and palm_yr>0:
                                a.append(str(palm_xr))
                                a.append(str(palm_yr))
                        # print("a=",a)
                        # print("len(a)",len(a))
                        # 描画
                        mp_drawing.draw_landmarks(
                            debug_image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        # debug_image = draw_hands_landmarks(
                        #     debug_image,
                        #     cx,
                        #     cy,
                        #     left_hand_landmarks,
                        #     # upper_body_only,
                        #     'R',
                        # )
                        # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    # 左手
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
                        ###找到手的坐标点
                        for index, landmark in enumerate(right_hand_landmarks.landmark):
                            if landmark.visibility < 0 or landmark.presence < 0:
                                    continue
                            palm_xl = min(int(landmark.x * cap_width), cap_width - 1)
                            palm_yl = min(int(landmark.y * cap_height), cap_height - 1)
                            if palm_xl>0 and palm_yl>0:
                                b.append(str(palm_xl))
                                b.append(str(palm_yl))
                        # print("b=",b)
                        # print("len(b)",len(b))

                        #关键点可视化
                        mp_drawing.draw_landmarks(
                            debug_image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        # debug_image = draw_bounding_rect(use_brect, debug_image, brect)

                    features1 = [] #单臂的特征点
                    features2=  [] #双臂的特征点

                    #判断是否左臂完全识别到，识别到了就进行单臂的动作分类
                    if len(b)==48 and len(a)!=48: 
                        features1.append(b)
                        logging.info("单臂分类")
                        if len(buffer_window1) < WINDOW_SIZE:  # WINDOW_SIZE = 32
                            buffer_window1.append(features1)
                        else:
                            index=0
                            model_input = torch.Tensor(np.array(buffer_window1, dtype=np.float32))
                            # model_input = torch.unsqueeze(model_input, dim=0)  # tensor(1,32,34)
                            model_input=model_input.permute(1,0,2)
                            y_pred = lstm_classifier1(model_input)
                            # print(y_pred)
                            prob = F.softmax(y_pred, dim=1)  ####softmax操作
                            # print(prob)
                            pred_index = prob.data.max(dim=1)[1]  ##tensor([i]) i为动作的索引值
                            logging.info(pred_index)

                            # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                            buffer_window1.pop(0)
                            buffer_window1.append(features1)
                            label = LABELS1[pred_index.numpy()[0]]
                            index=pred_index.numpy()[0]+1
                            # print(index)

                            l = label
                            # print("Label detected ", l)
                
                    
                    #判断左臂和右臂是否完全识别到，完全识别到了才会执行双臂的动作分类
                    elif len(a)==48 and len(b)==48:
                        features2.append(b+a)
                        logging.info("双臂分类")
                        l="机器人前进"
                    #     if len(buffer_window2) < WINDOW_SIZE:  # WINDOW_SIZE = 32
                    #         buffer_window2.append(features1)
                    #     else:
                    #         index=0
                    #         model_input = torch.Tensor(np.array(buffer_window2, dtype=np.float32))
                    #         # model_input = torch.unsqueeze(model_input, dim=0)  # tensor(1,32,34)
                    #         model_input=model_input.permute(1,0,2)
                    #         y_pred = lstm_classifier2(model_input)
                    #         # print(y_pred)
                    #         prob = F.softmax(y_pred, dim=1)  ####softmax操作
                    #         # print(prob)
                    #         pred_index = prob.data.max(dim=1)[1]  ##tensor([i]) i为动作的索引值
                    #         print(pred_index)

                    #         # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    #         buffer_window2.pop(0)
                    #         buffer_window2.append(features1)
                    #         label = LABELS2[pred_index.numpy()[0]]
                    #         index=pred_index.numpy()[0]+1
                    #         # print(index)

                    #         l = label
                    #         # print("Label detected ", l)
                    else:
                        index=0
                        l="未识别到动作"
                        logging.info("未识别到动作")


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
                    data['result']=index
                    # print(index)
                    if data is not None:
                        msg = {"robot": 2, "result": default_dump(index)}
                        gesture.sendMsg(3151, msg)
                        #Send_data(data_socket,1024,json.dumps(data,ensure_ascii=False,default=default_dump))
                    if open_gesture==0:
                        device.close()
                        cv.destroyAllWindows()
                        break
                    if open_gesture not in [0,1]:
                        print("无效的开启关闭标志位，请重新输入：")
                    # print(index)
    # cap.release()
    cv.destroyAllWindows()
def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
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
        # if index == 5:  # 人差指：付け根
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 9:  # 中指：付け根
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 13:  # 薬指：付け根
        #     palm_array = np.append(palm_array, landmark_point, axis=0)
        # if index == 17:  # 小指：付け根
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
#     # キーポイント
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
#         if index == 2:  # 親指：付け根
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 3:  # 親指：第1関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 4:  # 親指：指先
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 5:  # 人差指：付け根
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 6:  # 人差指：第2関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 7:  # 人差指：第1関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 8:  # 人差指：指先
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 9:  # 中指：付け根
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 10:  # 中指：第2関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 11:  # 中指：第1関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 13:  # 薬指：付け根
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 14:  # 薬指：第2関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 15:  # 薬指：第1関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 16:  # 薬指：指先
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#             cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
#         if index == 17:  # 小指：付け根
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 18:  # 小指：第2関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 19:  # 小指：第1関節
#             cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
#         if index == 20:  # 小指：指先
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
#     # 接続線 连接线
#     if len(landmark_point) > 0:
#         # 親指 拇指
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
#         # 薬指 无名指
#         cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
#         cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
#         cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)
#
#         # 小指
#         cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
#         cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
#         cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)
#
#         # 手の平 棕榈
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
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 0, 255), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 0, 255), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
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
        if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

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
def handlerMsg(msg_id, json_msg):
    # 打开/关闭手势识别
    global open_gesture
    if msg_id == 3100: 
        # 在此处添加消息处理逻辑
        # xxxxx（具体逻辑由手势识别程序实现)
        open_gesture = json_msg['start']
        print(open_gesture)
        # robot_id = json_msg['robot']
    else:
        print(json_msg)
        return
class Task:
    def __init__(self):
        super().__init__()
        gesture.registerHandler(handlerMsg)
    '''
    def run(self):
        global data_json
        global data_socket
        # data_socket=Client_connect()

        while True:
            time.sleep(1)
            #biaozhiwei=data_socket.recv(1024).decode("utf8")
            #print("标志位：",biaozhiwei)
            data_json['start']=json.loads(biaozhiwei)['start']
            # data_json = json.loads( biaozhiwei )
            if data_json['start']==1:
                print("收到开启标志位，开启手势识别")
            elif data_json['start']==0:
                print("收到关闭标志位，关闭手势识别")
    '''
def Exit(signum, data):
    global isExit
    gesture.exit()
    isExit = True

signal.signal(signal.SIGINT, Exit)

if __name__ == '__main__':
    #手势识别结果
    # #0表示没识别到结果
    # #1表示机器人停止
    # #2表示机器人右转
    # #3表示机器人左转
    # #4表示机器人后退
    # #5表示机器人前进
    data2={"robot":1,"id":3151,"result":0} 
    main(data2)
