import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import serial



window = tk.Tk()

window.title("Dr. Detection")

window.geometry("500x510")
window.configure(background ="lightgreen")

title = tk.Label(text="Click below to choose picture for testing track....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()

from twilio.rest import Client

# Find these values at https://twilio.com/user/account
account_sid = "AC2cec66adc7b7fb788b27f4a0e1052d8d"
auth_token = "329e1cabe7eb8d8aa534da00145d162c"

client = Client(account_sid, auth_token)

import serial
data = serial.Serial(
                'COM3',
		baudrate = 9600,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1 # must use when using data.readline()
                )

def send_data():
    print('sending')
    data.write(str.encode('C'))
    client.api.account.messages.create(
                        to="+91-9926783433",
                        from_="+18303965757",
                        body="Crack found")
    
def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    print("path " + verify_dir)
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'Railwaytrackcrack-{}-{}.model'.format(LR, '2conv-basic')
##    MODEL_NAME='keras_model.h5'
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data
##    def Send():
##        data.write(str.encode('C'))
##        print('Sent the Character')

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))


        if np.argmax(model_out) == 0:
            str_label = 'Broken'
            print('Broken')
        elif np.argmax(model_out) == 1:
            str_label = 'Non defective'
            print('Non defective')
        




        if str_label == 'Broken':
            track_type = "Broken track  "
            track = tk.Label(text='Analysed report : ' + track_type, background="lightgreen",
                               fg="Black", font=("", 15))
            track.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Broken track found', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            
            send_data()
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)
            
            
        elif str_label == 'Non defective':
            track_type = "BrownRotApple"
            track = tk.Label(text='Analysed report: ' + track_type, background="lightgreen",
                               fg="Black", font=("", 15))
            track.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Non Defective Track ', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            send_data('F')
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\deven\\Downloads\\CNN\\CNN\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')



        
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()



