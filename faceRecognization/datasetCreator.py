# DETECTOR
import cv2
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
video_capture=cv2.VideoCapture(0);
id=input('ENTER USER ID')
sampleNum = 0
while True:
    rect, img = video_capture.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces :
         sampleNum = sampleNum + 1
         cv2.imwrite('dataSet/user.'+str(id)+'.'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])    
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)    
         cv2.waitKey(100);
    cv2.imshow('Face',img);
    cv2.waitKey(1);
    
    if(sampleNum>20):
       break
video_capture.release()
cv2.destroyAllWindows()

    
