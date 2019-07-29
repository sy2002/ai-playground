#!/usr/bin/env python3

HC_PORT = '/dev/cu.usbserial-DN050L1O'

import readchar
import serial
import threading
import time

try:
    ser = serial.Serial(    port=HC_PORT,
                            baudrate=115200,
                            bytesize=8, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                            rtscts=False,
                            timeout=0)
except:
    print("ERROR: SERIAL PORT CANNOT BE OPENED.")
    sys.exit(1)

RUN_PROGRAM = True

def action_thread(ser):
    while RUN_PROGRAM:
        ser.write("D0".encode("ASCII"))
        time.sleep(0.1)
        print("d0")
        ser.write("d0".encode("ASCII"))


# while RUN_PROGRAM:
#     ch = readchar.readchar()
#     if ord(ch) == 17:
#         RUN_PROGRAM = False
#     send_str = ch.encode("ASCII")
#     ser.write(send_str)

def send_cmd(cmd):
    for ch in cmd:
        ser.write(ch.encode("ASCII"))
        time.sleep(0.8)

print("RESET")
send_cmd("x")

print("D0")
send_cmd("D0")

ch = readchar.readchar()

print("d0")
send_cmd("d0")

#t1 = threading.Thread(target=action_thread, args=[ser])
#t1.start()
#ch = readchar.readchar()
#ser.write("d0".encode("ASCII"))
#RUN_PROGRAM = False
#t1.join()
ser.close()
print("\n\nCONNECTION CLOSED!\n")
