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
from tkinter import *
import datetime
import cv2
import dlib
from imutils import face_utils
import record
import head_algo
from distractionModel.util.analysis_realtime import analysis




import tkinter
from tkinter import *
from PIL import Image, ImageTk










face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]





def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle



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

if (not(os.path.isfile("StudentDetails/StudentDetails.csv")) ):
    col = ["Id","Name"]
    with open('StudentDetails/StudentDetails.csv','a+',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(col)


img = PhotoImage(file="final.png")
label = Label(window,image=img)
label.place(x=0,y=0)








lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="white"  ,bg="#2a1b3d" ,font=('times', 15, ' bold ') )
lbl.place(x=100, y=430)

txt = tk.Entry(window,width=20   ,bg="#e1fcff" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=400, y=430,width=200,height=50)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="white"  ,bg="#2a1b3d"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=100, y=530)

txt2 = tk.Entry(window,width=20  ,bg="#e1fcff"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=400, y=530,width=200,height=50)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="white"  ,bg="#2a1b3d"  ,height=2 ,font=('times', 15, ' bold '))
lbl3.place(x=100, y=630)

message = tk.Label(window, text="" ,bg="#e1fcff"  ,fg="black"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
message.place(x=400, y=630)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="white"  ,bg="#004043"  ,height=2 ,font=('times', 15, ' bold  '))
lbl3.place(x=1000, y=630)


message2 = tk.Label(window, text="" ,bg="#e1fcff" ,fg="black",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold '))
message2.place(x=1300, y=630)

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

def TakeImages():
                             #imges are captured
    flag=False
    search_file = open("StudentDetails/StudentDetails.csv","rt")
    reader = csv.reader(search_file,delimiter=",")
    Id=(txt.get())   #id taken from user
    name=(txt2.get())
    #print(type(Id))
    for row in reader:
        if Id == row[0]:

            clear()
            clear2()
            res = "Id " + Id + " already Present"
            message.configure(text= res)
            flag=True
            break




    if flag == False:
        row = [Id , name]

        with open('StudentDetails/StudentDetails.csv','a+',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        csvFile.close()
        res = "Images Saved for ID : " + Id +" Name : "+ name
        #print(res)


        message.configure(text= res)


        if(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    sampleNum=sampleNum+1
                    cv2.imwrite('TrainingImage/'+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

                    cv2.imshow('frame',img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum>150:
                    break
            cam.release()
            cv2.destroyAllWindows()
            res = "Images Saved for ID : " + Id +" Name : "+ name
        else:
            if(is_number(Id)):
                res = "Enter Alphabetical Name"
                message.configure(text= res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text= res)



def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)

    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def conce():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)

    #cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    sum1,sum2,sum3=0,0,0
    d={'left':0,'right':0,'up':0,'down':0,'center':0}
    count = 0
    compare_time= datetime.datetime.now()
    compare_time= compare_time.strftime("%M")
    compare_time=int(compare_time)+1
    #print(h)
    active_count=0
    inactive_count=0
    x=0
    y=0
    row=1
    name=[]
    l=[]
    l1=[]
    l2=[]
    maxi=[]
    eye_count=0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)

        ana1=analysis()
        eye=ana1.detect_face(frame)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                std_name=aa[0]
                tt=str(Id)+"-"+aa
                print()
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id='Unknown'
                std_name=Id
                tt=str(Id)

            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", frame[y:y+h,x:x+w])
            #name.append(str(tt))
            cv2.putText(frame,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        #cv2.imshow('frame',frame)

        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                x=euler_angle[0, 0]
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                y=euler_angle[1, 0]
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                z=euler_angle[2, 0]
                if x<0 and y<0 and z>0:
                    d['up']+=1
                elif y<0 and z<0 and x>0:
                    d['left']+=1
                elif x<0 and z<0 and y>0:
                    d['right']+=1
                elif x>0 and y>0 and z>0:
                    d['down']+=1
                else:
                    d['center']+=1
            cv2.imshow("demo", frame)
            if  eye_count <= 1 and (x>=-20 and x<=20) and (y>=-20 and y<=20):
               active_count+=1
               if eye==0:
                  eye_count+=1
               else:
                  eye_count=0
            else:
               inactive_count+=1
               if eye==0:
                    eye_count+=1
               else:
                    eye_count=0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(name)
                print(l)
                print(l1)
                print(l2)
                for k,v in d.items():
                    print(k)
                    print(v)
                record.record1(name,l,l1,l2)
                break
        repeat = datetime.datetime.now()
        h2="he is active for 1 min"
        h3="he is inactive for 1 min"
        if repeat.strftime("%M") == str(compare_time):
           print(count)
           print("hii")
           if active_count >= inactive_count:
               print("he is active for 1 min")
               l.append(h2)
           else:
               print("he is inactive for 1 min")
               l.append(h3)
           print(active_count,inactive_count)
           l1.append(active_count)
           l2.append(inactive_count)
           name.append(std_name)
           active_count=0
           inactive_count=0
           count=0
           compare_time+=1


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id='Unknown'
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    cdate = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")

    #row = attendance.values.tolist()
    today=datetime.date.today()


    if (str(today)==str(cdate)):                               #not((cdate>today) or (today>cdate))
        print(attendance)
        print(type(attendance))

        rec_id=[]
        fileName="Attendance/Attendance_"+cdate+".csv"
        # search_file = open(fileName,"rt")
        # reader = csv.reader(search_file,delimiter=',')
        # for row in reader:
        #     rec_id.append(row[0])
        # print(*rec_id)


        #fileName="Attendance\Attendance_"+cdate+".csv"
        if (not(os.path.isfile(fileName)) ):                        #1st time creation of file
            attendance.to_csv(fileName,index=False)

            # with open(fileName,'a+',newline='') as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(row)
               # attendance.to_csv(fileName,index=False)
        else:                                                   #Append data
            # with open(fileName,'a+',newline='') as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(row)

            flag=False
            search_file = open(fileName,"rt")
            reader = csv.reader(search_file,delimiter=',')
            for row in reader:
                if Id==row[0]:
                    flag=True
                    print(row[0])
                    break
            if (flag==False):
                attendance.to_csv(fileName,mode='a',header=False,index=False)





    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
# fucntion to check the work hour of the person
def checkWork():
    Id = txt.get()
    ts = time.time()
    cdate = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') # take the current datetime
    fileName = "Attendance/Attendance_" + cdate + '.csv' # path of the file to be searched in
    search_file = open(fileName, 'rt')
    listtime = [] # list of the timimg of every recognition occurance of the given ID
    reader = csv.reader(search_file, delimiter=',')
    noWorkFlag = True # flag variable to check its work-time
    for row in reader:
        if Id == row[0]:
            listtime.append(row[3]) # append the timimgs of the matched the data
            noWorkFlag = False
    if noWorkFlag: # if noWorkFalg is false means no work data is present
        message2.configure(text='No Work Record for today')
    if not noWorkFlag: # if the noWorkFlag is true then notify their work-hour data
        datetime1 = datetime.datetime.strptime(listtime[0], '%H:%M:%S')
        datetime2 = datetime.datetime.strptime(listtime[-1], '%H:%M:%S')
        str1 = str(datetime2 - datetime1)
        message2.configure(text='Hr:Min:Sec = ' + str1)

clearButton = tk.Button(window,text="Clear", command=clear  ,fg="white"  ,bg="#25274d",activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=700, y=430,height=25,width=100)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="white"  ,bg="#25274d"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=700, y=530,height=25,width=100)
takeImg = tk.Button(window, text="Capture Faces", command=TakeImages  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=100, y=800)
trainImg = tk.Button(window, text="Train Faces", command=TrainImages  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=400, y=800)
trackImg = tk.Button(window, text="Recognize Faces", command=TrackImages  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=1000, y=430)
quitWindow = tk.Button(window, text="Exit", command=window.destroy  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1400, y=800)

head = tk.Button(window, text="Concentration", command=conce  ,fg="white"  ,bg="#660066"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
head.place(x=1400,y=430)

copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
atte = tk.Button(window, text="Attendace", command=checkWork  ,fg="white"  ,bg="#660066",width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
atte.place(x=1000, y=800)
window.mainloop()
