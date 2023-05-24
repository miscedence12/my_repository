#-*-coding:utf-8-*-
import socket
import json
from threading import Thread
from tkinter import N
from tkinter.messagebox import NO
import time
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def main():
    ip_port=('127.0.0.1', 10086)
    #TCP
    sk=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sk.bind(ip_port)
    sk.listen(5)
    print("服务端启动,地址为：{}；端口为：{}....".format(ip_port[0],ip_port[1]))
    #sock传输数据
    sock,addr=sk.accept()
    print("客户端连接成功")
    # biaozhiwei=input("请往客户端发送标志位：")
    # sock.send(biaozhiwei.encode("utf8")) 
    # sock.close()
    number=1
    global data
    while True:
        # print("recepting.......")
        #获取从客户端发送来的数据str类型
        data=sock.recv(1024)    
        # if not data:
        #     break
        #接收到的客户端数据
        str_data=data.decode("utf8")

        #给客户端返回的数据
        msg="服务端返回的数据:"+str_data
        if str_data=="未识别到动作" and number==1:
            sock.send(msg.encode("utf8"))
            number=2
            # print(number)
            print("接收到数据：未识别到动作")
            
        elif str_data!="未识别到动作":
            sock.send(msg.encode("utf8"))
            number=1
            # print(number)
            print("接收到数据:",str_data)
  
def main1():
    # 创建udp套接字,
    # AF_INET表示ip地址的类型是ipv4，
    # SOCK_DGRAM表示传输的协议类型是udp
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
    # 绑定本地信息，若不绑定，系统会自动分配
    bind_addr = ("127.0.0.1", 10085)
    udp_socket.bind(bind_addr) # ip和port，ip一般不用写，表示本机的任何一个ip
 
    while True:
        # 等待接收数据
        print("等待接收数据:..........")
        revc_data  = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数

        # 打印接收到的数据
        data=revc_data[0].decode("utf8")
        print(data)
        if data=="机器人停止":
            pass
        # 给客户端返回的数据
    # 关闭套接字
    # udp_socket.close()
def one_servicer():
    #Create The Socket
    global sock
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    #Listen The Port
    s.bind(('127.0.0.1',10085))
    s.listen(5)
    print('Waiting for connection...')
    t1=Task()
    t1.start()
    def tcplink(sock,addr):
        print('\nAccept new connection from %s:%s...' % addr)
        while True:
            data=sock.recv(1024).decode()
            # if data is not None:
            #     print("data: ", data)
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
                    print("得到反馈成功的信号")
                if data_json['ack']==0:
                    print("得到反馈失败的信号")

        sock.close()
        print('Connection from %s:%s closed.'%addr)

        
    while True:
        # 开始一个新连接
        sock,addr=s.accept()
        # 创建一个线程来处理连接
        # t=threading.Thread(target=tcplink(sock,addr))
        tcplink(sock,addr)

class Task(Thread):
    def __init__(self):
        super().__init__()
    def run(self): 
        while True:
            #主控到手势识别
            time.sleep(1)
            data_kaiguan={'robot':1,'id':3199,'start':0}  #0表示关闭手势识别 1表示开启手势识别
            data_kaiguan['start']=int(input("请输入手势开启或关闭标志位："))
            data_kaiguan=json.dumps(data_kaiguan)
            sock.send(data_kaiguan.encode() )
if __name__ == "__main__":
    one_servicer()
 