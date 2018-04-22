import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path ='dataSet'
#mthod to get Images and Ids
def getImageId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces =[]
    IDs =[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        facesNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        #storing
        faces.append(facesNp)
        IDs.append(ID)
        cv2.imshow("traning",facesNp)
        cv2.waitKey(10)
    return np.array(IDs),faces
Ids,faces = getImageId(path)
recognizer.train(faces,Ids)
recognizer.save('recognizer/tranningData.yml')
cv2.destroyAllWindows()