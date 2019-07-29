
HC_PORT = '/dev/cu.usbserial-DN050L1O'

from readchar import readchar
import serial
import threading
from time import sleep
import sys

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
        sleep(0.01)                        # sleep 10ms, free CPU cycles, keep CPU usage low
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

#    readchar()
#    ser.write("?".encode("ASCII"))    

    toggle = False
    i = 0
    chars = "D0d0"

    global CONTINUE_RUNNING
    while CONTINUE_RUNNING:
        #ch = readchar()                         # blocking call
        ch = "a"

        if ord(ch) == 17:
            CONTINUE_RUNNING = False
        else:
            #print(ch, end="")                       # echo all typed characters
            sys.stdout.flush()                      # see the characters instantly
            try:
                #send_str = ch.encode("ASCII")       # convert python string to byte stream
                send_str = chars[i].encode("ASCII")
                #print(send_str)
                ser.write(send_str)                 # send non-blocking due to "timeout=0" in serial.Serial(...)            
                i = (i + 1) % 4
            except:
                pass

def do_thread(ser):
    global CONTINUE_RUNNING
    while CONTINUE_RUNNING:
        ser.write("i".encode("ASCII"))
        sleep(0.8)
        ser.write("o".encode("ASCII"))
        sleep(0.8)
        ser.write("h".encode("ASCII"))
        sleep(0.8)


received = ""
while (received != "RESET\n"):
    print("Reset attempt")
    ser.write("x".encode("ASCII"))
    sleep(1.0)
    amount = ser.in_waiting
    if amount > 0:
        ser_in = ser.read(amount)           # due to "timeout=0" this is a non-blocking serial read
        received = ser_in.decode("ASCII")
        print("Received: ", received)


# The serial interface is full duplex and therefore reading and writing operations occur concurrently.
# "join()" means: wait until the thread ends
t1 = threading.Thread(target=read_thread, args=[ser])
t2 = threading.Thread(target=write_thread, args=[ser])
#t3 = threading.Thread(target=do_thread, args=[ser])
t1.start()
t2.start()
#t3.start()
t1.join()
t2.join()
#t3.join()
ser.close()
print("\n\nCONNECTION CLOSED!\n")
