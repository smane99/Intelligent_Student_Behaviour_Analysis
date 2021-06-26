import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from datetime import date
#import PIL.Image

import datetime
import cv2
import dlib
from imutils import face_utils
import record
import head_algo
from distractionModel.util.analysis_realtime import analysis
import pymysql
import mysql.connector
import insert
import read2

from tkinter import*
window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("INTELLIGENT STUDENT BEHAVIOUR ANALYSIS AND ATTENDANCE SYSTEM")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)

#window.geometry('1280x720')
window.configure(background='#97CAEF')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)



lbl = tk.Label(window, text="Enter Username",width=20  ,height=2  ,fg="white"  ,bg="#2a1b3d" ,font=('times', 15, ' bold ') )
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20   ,bg="#e1fcff" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=700, y=200,width=200,height=50)

lbl2 = tk.Label(window, text="Enter Password",width=20  ,fg="white"  ,bg="#2a1b3d"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="#e1fcff"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=300,width=200,height=50)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="white"  ,bg="#2a1b3d"  ,height=2 ,font=('times', 15, ' bold '))
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="#e1fcff"  ,fg="black"  ,width=60  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
message.place(x=700, y=400)

#lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="white"  ,bg="#004043"  ,height=2 ,font=('times', 15, ' bold  '))
#lbl3.place(x=400, y=650)


#message2 = tk.Label(window, text="" ,bg="#e1fcff" ,fg="black",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold '))
#message2.place(x=700, y=650)

#img = PhotoImage(file="log1.png")
#label = Label(window,image=img)
#label.place(x=0,y=0)





def imp():

    Username=txt.get()
    Password=txt2.get()
    f=read2.read3(Username,Password)
    if f==1:
        os.system('python train_main.py')
        res="Welcome!!!"
        message.configure(text=res)
    else:
        print("Something went wrong!!!")
        res1="Invalid Credentials or Register your Credentials"
        message.configure(text=res1)


def add1():
    #os.system('python register.py')

    Username=txt.get()
    Password=txt2.get()
    insert.ins(Username,Password)

def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text= res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False




#clearButton = tk.Button(window, text="Clear", command=clear  ,fg="white"  ,bg="#25274d"   ,activebackground = "Red" ,font=('times', 15, ' bold '))
#clearButton.place(x=950, y=215,height=25,width=100)
#clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="white"  ,bg="#25274d"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
#clearButton2.place(x=950, y=312,height=25,width=100)
Login = tk.Button(window, text="Login",fg="white",command=imp,bg="#660066"  ,borderwidth = 10,width=20  ,height=3, activebackground = "Green" ,font=('times', 15, ' bold '))
Login.place(x=400, y=500)
register = tk.Button(window, text="Register", command=add1  ,fg="white"  ,bg="#660066",borderwidth = 10,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
register.place(x=700, y=500)
#trackImg = tk.Button(window, text="Recognize Faces", command=TrackImages  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
#trackImg.place(x=800, y=500)
#quitWindow = tk.Button(window, text="Exit", command=window.destroy  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
#quitWindow.place(x=1400, y=500)

#head = tk.Button(window, text="Concentration", command=conce  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
#head.place(x=1100,y=500)

#copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
#copyWrite.tag_configure("superscript", offset=10)

window.mainloop()
