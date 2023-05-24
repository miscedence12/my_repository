# #########################################
# @Author: jipp 
# @Date: 2022-10-22 17:20:06 
# @Last Modified by:   jipp 
# @Last Modified time: 2022-10-22 17:20:06 
# #########################################
import json

SUNLINK_HEADER_ONE = 0x42
SUNLINK_HEADER_TWO = 0x53
SUNLINK_HEADER_THR = 0x03

def checkCrc(data, size):
    gen = 0xA001
    crc = 0xFFFF

    if data == None:
        return crc
    
    for i in range(size):
        crc ^=  data[i]
        for k in range(8):
            if (crc & 0x01) == 0x01:
                crc = crc >> 1
                crc = crc ^ gen
            else:
                crc = crc >> 1
    return crc

def intToChar(data):
    return data.to_bytes(length=1, byteorder='little').decode('UTF-8')

def wrapper(msg_id, json_msg):
    # 转化成string
    msg = json.dumps(json_msg)

    buf = bytearray()
    buf.append(SUNLINK_HEADER_ONE)
    buf.append(SUNLINK_HEADER_TWO)
    buf.append(SUNLINK_HEADER_THR)
    
    # 2 --> size
    str_len = len(msg)
    msg_len = str_len + 2 + 4

    # 记录消息长度
    buf.append((msg_len & 0x00FF))
    buf.append((msg_len >> 8))

    # 记录消息ID
    buf.append((msg_id & 0x00FF))
    buf.append((msg_id >> 8))

    # 序列化字符串尺寸
    buf.append((str_len & 0x00FF))
    buf.append((str_len >> 8))

    # 拷贝数据
    for item in msg:
        buf.append(int.from_bytes(item.encode('UTF-8'), byteorder='little'))

    # CRC校验
    msg_crc = checkCrc(buf, len(buf))
    buf.append(msg_crc & 0xFF)
    buf.append(msg_crc >> 8)
    return buf

# 从字节流中将数据转换乘对应的消息内容（转换成‘列表’）
class ParseMsg:
    buf = None

    def __init__(self):
        self.buf = []

    def pushData(self, data):
        for item in data:
            self.buf.append(item)

    def parseMsg(self):
        msg = []
        if len(self.buf) < 7:
            return False, -1, json.loads('{}')
        
        # print(self.buf)

        # 判断消息头是否正确
        if not self.msgHeader(self.buf):
            return False, -1, json.loads('{}')

        # print('Msg Header Ok')
        # 获取消息长度
        msg_len = (self.buf[4] << 8)  + self.buf[3] - 4

        # print('Msg Len: ', msg_len)
        # 判断消息接受是否完整
        if len(self.buf) < msg_len + 9:
            return False, -1, json.loads('{}')

        # 进行数据校验
        msg_crc = (self.buf[msg_len + 8] << 8) + self.buf[msg_len + 7] 
        new_crc = checkCrc(self.buf, msg_len + 7)
        # print('old crc:', hex(msg_crc), 'new_crc:', hex(new_crc))
        if new_crc != msg_crc:
            self.popByte(0, msg_len + 9)
            return False, -1, json.loads('{}')
        # print('Old: ', msg_crc, ', New: ', new_crc)
        
        # 获取消息id
        msg_id = self.buf[5] + (self.buf[6] << 8)
        str_size = self.buf[7] + (self.buf[8] << 8)
        # print('MsgID: ', msg_id, 'StrSize: ', str_size)

        # 删除头部数据
        self.popByte(0, 9)

        # 获取消息
        for i in range(0, str_size):
            msg.append(intToChar(self.buf[i]))
        
        # 删除消息(2 --> crc)
        self.popByte(0, str_size + 2)

        try:
            json_data = json.loads(''.join(msg))
        except:
            print('unicode decode error')
            return False, msg_id, json.loads('{}')

        return True, msg_id, json_data

    # 校验消息头
    def msgHeader(self, data):
        if data == None:
            return False
        
        header1 = data[0]
        header2 = data[1]
        header3 = data[2]

        if header1 != SUNLINK_HEADER_ONE:
            print('header1: ', header1)
            return False

        if header2 != SUNLINK_HEADER_TWO:
            print('header2: ', header2)
            return False

        if header3 != SUNLINK_HEADER_THR:
            print('header3: ', header3)
            return False

        return True

    def popByte(self, start, len):
        for i in range(start, start + len):
            self.buf.pop(start)