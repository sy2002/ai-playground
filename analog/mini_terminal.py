#!/usr/bin/env python3

# Mini Terminal for connecting to Model-1 by Analog Paradigm
#
# Setup:
#   1. Install dependencies: pip install -r mini_terminal.txt
#   2. Set the HC_PORT variable (below)
#   3. Run via python3 mini_terminal.py or by directly executing ./mini_terminal.py
#
# Exit using CTRL+Q
#
# done by sy2002 on 27th of July 2019

# on macOS and Linux enter "ll -l /dev/cu*" in terminal to find out where to connect to
# the following port is the one on sy2002's computer; you'll probabily need to adjust it to yours
HC_PORT = '/dev/cu.usbserial-DN050L1O'

import readchar
import serial
import threading
import time
import sys

print("\nHC Mini Terminal, done by sy2002 on 27th of July 2019")
print("=====================================================\n")
print("press CTRL+Q to exit\n")

try:
    ser = serial.Serial(    port=HC_PORT,
                            baudrate=115200,
                            bytesize=8, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                            rtscts=False,
                            timeout=0)
except:
    print("ERROR: SERIAL PORT CANNOT BE OPENED.")
    sys.exit(1)

CONTINUE_RUNNING = True                         # if set to False, the threads end

def read_thread(ser):
    global CONTINUE_RUNNING
    while CONTINUE_RUNNING:
        time.sleep(0.01)                        # sleep 10ms, free CPU cycles, keep CPU usage low
        amount = ser.in_waiting                 # amount of chars in serial read buffer
        input_str = ""
        if amount > 0:
            ser_in = ser.read(amount)           # due to "timeout=0" this is a non-blocking serial read
            input_str = ser_in.decode("ASCII")  # interpret byte stream as ASCII and convert to python string
            # for some reason, we need to convert \n into \r to have proper screen output
            for c in input_str:
                if c == "\n":
                    print("\r")
                else:
                    print(c, end="")
            sys.stdout.flush()                  # necessary to make sure we see the printed strint immediatelly

def write_thread(ser):
    global CONTINUE_RUNNING
    while CONTINUE_RUNNING:
        ch = readchar.readchar()                # blocking call
        print(ch, end="")                       # echo all typed characters
        sys.stdout.flush()                      # see the characters instantly
        try:
            send_str = ch.encode("ASCII")       # convert python string to byte stream
            ser.write(send_str)                 # send non-blocking due to "timeout=0" in serial.Serial(...)            
        except:
            pass
        
        if ord(ch) == 17:                       # CTRL+Q ends the program
            CONTINUE_RUNNING = False

# The serial interface is full duplex and therefore reading and writing operations occur concurrently.
# "join()" means: wait until the thread ends
t1 = threading.Thread(target=read_thread, args=[ser])
t2 = threading.Thread(target=write_thread, args=[ser])
t1.start()
t2.start()
t1.join()
t2.join()
ser.close()
print("\n\nCONNECTION CLOSED!\n")
