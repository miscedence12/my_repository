#!/usr/bin/python3
#-*-coding:utf-8-*-
import socket
import json
from threading import Thread
import time
import logging
import signal
import rospy
from geometry_msgs.msg import Twist

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

isExit = False
client_flag=0  #定义一个标志位

rospy.init_node("move_robot",anonymous=True)#初始化结点
pub=rospy.Publisher("/turtle1/cmd_vel",Twist,queue_size=10)#创建一个发布者，向话题发布消息
rate=rospy.Rate(10)#10HZ
vel_msg=Twist() #创建一个Twist类型的消息

class Server:

    def create_socket(self):
        try:
            global host
            global port
            global s
            host = "192.168.31.51"
            port = 10085
            s = socket.socket()
        except socket.error as msg:
            self.create_socket()
    def bind_socket(self):
        try:
            global host
            global port
            global s
            s.bind((host, port))
            s.listen(5)
        except socket.error as msg:
            self.bind_socket()
        
    def one_servicer(self):
        global s
        #创建一个套接字
        self.create_socket()
        #监听端口
        self.bind_socket()
        print('服务器等待被连接...')
class Tcplink(Thread):
    def __init__(self):
        super().__init__()
    def run(self):
        global isExit
        global data
        global client_flag
        global sock
        global data_json
        logging.info("tcplink start")
        sock,addr=s.accept()
        logging.info('接收到地址为 %s:%s的客户端连接' %addr)
        client_flag=1
        t1=Task()
        t1.start()
        while not isExit:  
            data=sock.recv(1024).decode()
            print(type(data))

            if  not data:
                break
            try:
                data_json = json.loads(data,encoding="utf-8")
                print("data_loads",type(data_json))
                
            except json.decoder.JSONDecodeError as e:
                print("decode error,data:",data)


                continue
          
            
            if "result" in data_json:
                if data_json["result"] == 0 or data_json["result"] == 32: #没识别到结果
                    rospy.loginfo("未识别到动作,action:0")
                    print("0")
                
                if data_json["result"] == 1: #机器人停止
                    rospy.loginfo("机器人停止,action:1")
                    print("1")
                    vel_msg.linear.x=0
                    vel_msg.angular.z=0
                    pub.publish(vel_msg)
                    rate.sleep()
                
                    
                if data_json["result"] == 2: #机器人右转
                    print("2")
                    rospy.loginfo("机器人右转,action:2")
                    vel_msg.linear.x=0
                    vel_msg.angular.z=-0.2
                    pub.publish(vel_msg)
                    rate.sleep()
                
                
                if data_json["result"] == 3: #机器人左转
                    print("3")
                    rospy.loginfo("机器人左转,action:3")
                    vel_msg.linear.x=0
                    vel_msg.angular.z=0.2
                    pub.publish(vel_msg)
                    rate.sleep()
                
                
                if data_json["result"] == 4: #机器人后退
                    print("4")
                    rospy.loginfo("机器人后退,action:4")
                    vel_msg.linear.x=-0.2
                    vel_msg.angular.z=0
                    pub.publish(vel_msg)
                    rate.sleep()
                
                
                if data_json["result"] == 5: #机器人前进
                    print("5")
                    rospy.loginfo("机器人前进,action:5")
                    vel_msg.linear.x=0.2
                    vel_msg.angular.z=0
                    pub.publish(vel_msg)
                    rate.sleep()
                
                if data_json["result"] not in [0,1,2,3,4,5,32]:
                    rospy.loginfo("机器人无效的动作标志,action:none")
               
            # if "ack" in data_json:
            #     if data_json['ack']==1:
            #         print("得到反馈成功的信号")
            #     if data_json['ack']==0:
            #         print("得到反馈失败的信号")     
        sock.close()
        print('客户端 %s:%s 已断开，等待下一次连接.....'%addr)
   

class Task(Thread):
    def __init__(self):
        super().__init__()
    def run(self): 
        global sock
        while not isExit:
            #主控到手势识别
            time.sleep(1)
            data_kaiguan={'robot':1,'id':3199,'start':0}  #0表示关闭手势识别 1表示开启手势识别
            data_kaiguan['start']=int(input("请输入手势开启或关闭标志位："))
            # data_kaiguan['start']=1
            data_kaiguan=json.dumps(data_kaiguan)
            sock.send(data_kaiguan.encode())
            if isExit:
                break

def main():
    global client_flag
    server=Server()
    server.one_servicer()
    # 开始一个新连接
    t2=Tcplink()
    t2.start()
    while not isExit:
        if client_flag:
            t2=Tcplink()
            t2.start()
            client_flag=0
def Exit(signum,data):
    global isExit
    global client_flag
    client_flag=0
    isExit=True
    print("server exit successfully")
signal.signal(signal.SIGINT,Exit)

if __name__ == "__main__":
    main()
   