import serial
import time
data = serial.Serial(
                'COM5',
		baudrate = 9600,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1 # must use when using data.readline()
                )


while True :
    time.sleep(0.35)
    data.write(str.encode('C'))
    print('DOne')

    x=data.read(2)

    print('x {}'.format(x))
