#DETECTOR FILE
import cv2
#faceDetect 
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#Video Capture
video_capture = cv2.VideoCapture(0)
# recongnizer Import here
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\tranningData.yml')
#set id
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
#loop function
while True :
    rect,img = video_capture.read();
    #Convert to Gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray,1.3,5);
    #loop function
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #conf
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        
        if(conf<70):
            if(id==1) :
               id = "Suprim"
            if(id==4) :
               id = "Suray"
        else :
            id="unknow"
         #Putting Text
        cv2.putText(img,str(id),(x,y-20),font,3,(0,255,255),2)
    cv2.imshow('Face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
