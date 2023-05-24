#-*-coding:utf-8-*-
import socket
import json
from threading import Thread
from tkinter import N
from tkinter.messagebox import NO
import time
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

host="127.0.0.1"
port=10085
isExit = False
client_flag=0  #定义一个标志位

class Server:
    def create_socket(self):
        try:
            global host
            global port
            global s
            host = ""
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
            self.ind_socket()

    def one_servicer(self):
        global s
        #Create The Socket
        self.create_socket()
        #Listen The Port
        self.bind_socket()
        print('服务器等待被连接...')

    def tcplink(self,sock,addr):
        global isExit
        global data
        global client_flag
        print('接收到地址为 %s:%s的客户端连接' % addr)
        t1=Task()
        t1.start()
        while not isExit: 
            time.sleep(1)   
            data=sock.recv(1024).decode()
            # t=Target()
            # t.start()
            if  not data:
                break
            try:
                data_json = json.loads( data )
            except json.decoder.JSONDecodeError as e:
                continue
            if "result" in data_json:
                if data_json["result"] == 0 or data_json["result"] == 32:
                    # print("0")
                    pass
                elif data_json["result"] == 1:
                    # print("1")
                    pass
                elif data_json["result"] == 2:
                    # print("2")
                    pass
                    
                elif data_json["result"] == 3:
                    # print("3")
                    pass
                
                elif data_json["result"] == 4:
                    # print("4")
                    pass
                
                elif data_json["result"] == 5:
                    # print("5")
                    pass
                elif data_json["result"] not in [0,1,2,3,4,5,32]:
                    break
                # elif number==100:
                #     break
            if "ack" in data_json:
                if data_json['ack']==1:
                    logging.info("得到反馈成功的信号")
                if data_json['ack']==0:
                    logging.info("得到反馈失败的信号")
            
        sock.close()
        print('Connection from %s:%s closed.'%addr)

class Target(Thread):
    def __init__(self):
        super().__init__()
    def run(self):
        global isExit
        global data
        while not isExit:
            time.sleep(1)
            logging.info("在判断")
            if data==b"":
                logging.info("客户端断开连接")
                isExit=True  

class Task(Thread):
    def __init__(self):
        super().__init__()
    def run(self): 
        global sock
        while True:
            #主控到手势识别
            time.sleep(1)
            data_kaiguan={'robot':1,'id':3199,'start':0}  #0表示关闭手势识别 1表示开启手势识别
            data_kaiguan['start']=int(input("请输入手势开启或关闭标志位："))
            data_kaiguan=json.dumps(data_kaiguan)
            sock.send(data_kaiguan.encode() )


if __name__ == "__main__":
    server=Server()
    server.one_servicer()
    while True:
        global sock
        # 开始一个新连接
        sock,addr=s.accept()
        server.tcplink(sock,addr)
 