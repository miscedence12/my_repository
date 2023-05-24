import socket
import time
import logging
import json
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
def Client_connect():
    global dataSocket
    #  === TCP 客户端程序 client.py ===
    IP = '192.168.31.51'
    SERVER_PORT = 10085
    BUFLEN = 1024
    ended=False
    # 实例化一个socket对象，指明TCP协议
    dataSocket = socket.socket(socket.AF_INET,socket. SOCK_STREAM)
    while not ended:
        try:
            # 连接服务端socket
            dataSocket.connect((IP, SERVER_PORT))
        except:
            logging.warning('未发现服务器程序,两秒后尝试继续连接')
            time.sleep(2)
        else:
            break
    logging.info("TCP连接建立成功......")

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
def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
data_socket=Client_connect()

data2={"robot":1,"msg_id":3151,"result":0} 
while True:
    time.sleep(1)
    Send_data(data_socket,1024,json.dumps(data2,ensure_ascii=False,default=default_dump))