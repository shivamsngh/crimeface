import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
sdThresh = 10
cap = cv2.VideoCapture(0)
face_count = 0 
_, frame1 = cap.read()
_, frame2 = cap.read()
# cap.set(3,640) # set Width
# cap.set(4,480) # set Height
cv2.WINDOW_KEEPRATIO

def distMap(frame1, frame2):
     #"""outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist

while(True):
    ret, img = cap.read()
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)
    dist = distMap(frame1, frame3)
    frame1 = frame2
    frame2 = frame3
    mod = cv2.GaussianBlur(dist, (9,9), 0)
    _, thresh = cv2.threshold(mod, 100, 255, 0)
    _, stDev = cv2.meanStdDev(mod)
    cv2.imshow('dist', mod)
    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
    if stDev > sdThresh:
        print("Motion detected.. Do something!!!"); 
        # frame = cv2.flip(frame, -1) # Flip camera vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
         )
        face_count=len(faces)
        cv2.putText(img, 'Total Faces:'+str(face_count), (100, 100), font, 1, (255,255,255), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color=img[y:y+h, x:x+w]
    
        cv2.imshow('frame', img)
    # cv2.imshow('gray', gray)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()