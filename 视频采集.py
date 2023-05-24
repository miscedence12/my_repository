import cv2
import depthai as dai

def video_collect():
  cap = cv2.VideoCapture(0)
  w=960
  h=540
  size=(w,h)

  out = cv2.VideoWriter("./output/train/b2.avi",cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (w,h))

  while (cap.isOpened()):
    ret, frame = cap.read()
    # image = cv2.flip(frame, 1)
    if ret == True:
      frame=cv2.resize(frame,size)
      out.write(frame)
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break
  print("视频帧数:",cap.get(7))

  cap.release()
  out.release()
  cv2.destroyAllWindows()

def Y_collect() :
  a=[]
  f = open("./output/Y_test.txt", "a", encoding="utf-8")
  for i in range(0,28):
    a.append(str(1)+"\n")
  #
  for i in range(0,24):
    a.append(str(2)+"\n")

  for i in range(0,24):
    a.append(str(3)+"\n")

  for i in range(0,22):
    a.append(str(4)+"\n")

  # for i in range(0,22):
  #   a.append(str(5)+"\n")

  f.writelines(a)
  f.close()
def Split_32():
  cap = cv2.VideoCapture("output/HRG/train/b2.avi")
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
  frames_num=cap.get(7)
  print("视频总帧数：",frames_num)
  fram_int=frames_num//32
  print("32帧倍数",fram_int)
  a=0
  while True:
    ret, image = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1)
    a+=1
    if key == 27:  # ESC
      break
    cv2.imshow("show",image)
    if a==int(fram_int*32):
      break
  print("a:",a)
  cv2.destroyAllWindows()

def depthai_video():
  pipeline=dai.Pipeline()
  # 定义相机源和输出
  camRgb = pipeline.create(dai.node.ColorCamera)
  xoutRgb = pipeline.create(dai.node.XLinkOut)

  xoutRgb.setStreamName("rgb")

  # 属性
  camRgb.setPreviewSize(960, 540)
  camRgb.setInterleaved(False)
  camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

  # 连接
  camRgb.preview.link(xoutRgb.input)
  #视频大小
  w=960
  h=540
  size=(w,h)
  out = cv2.VideoWriter("./tree/1.avi",cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (w,h))
  #打开相机
  with dai.Device(pipeline) as device:
    print('Connected cameras: ', device.getConnectedCameras())
    # 输出usb传输速度
    print('Usb speed: ', device.getUsbSpeed().name)
    #输出序列来得到rgb图像
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
      inRgb = qRgb.get()  
      # 相机捕捉 
      image = inRgb.getCvFrame()
      frame=cv2.resize(image,size)
      out.write(frame)
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    device.close()
         





if __name__=="__main__":
  # video_collect()
  # Y_collect()
  # Split_32()
  depthai_video()
 



