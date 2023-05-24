import socket

def Client():
    #  === TCP 客户端程序 client.py ===
    IP = '192.168.171.10'
    SERVER_PORT = 10086
    BUFLEN = 1024

    # 实例化一个socket对象，指明TCP协议
    dataSocket = socket.socket(socket.AF_INET,socket. SOCK_STREAM)

    # 连接服务端socket
    dataSocket.connect((IP, SERVER_PORT))

    while True:
        # 从终端读入用户输入的字符串
        toSend = input('请输入需要发送的信息：')
        if  toSend =='exit':
            break
        # 发送消息，也要编码为 bytes
        dataSocket.send(toSend.encode("utf8"))

        # 等待接收服务端的消息
        recved = dataSocket.recv(BUFLEN)
        # 如果返回空bytes，表示对方关闭了连接
        if not recved:
            break
        # 打印读取的信息
        print(recved.decode())

    dataSocket.close()
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

# def Send_data(dataSocket,BUFLEN,send_data):
#     # 从终端读入用户输入的字符串
#     toSend = send_data
#     # 发送消息，也要编码为 bytes
#     dataSocket.send(toSend.encode("utf8"))

#     # 等待接收服务端的消息
#     #非阻塞模式
#     dataSocket.setblocking(0)
#     try:
#         recved = dataSocket.recv(BUFLEN)
#         # 打印读取的信息
#         print(recved.decode())
#     except BlockingIOError as e:
#         #阻塞模式
#         dataSocket.setblocking(1)
#     # 如果返回空bytes，表示对方关闭了连接
 
    

    # dataSocket.close()
def Tset():
    data_socket=Client_connect()
    biaozhiwei=data_socket.recv(1024).decode("utf8")
    print("标志位：",biaozhiwei)
if __name__ =="__main__":
    Tset()