# Importing all necessary libraries 
import cv2 
import os 
from retinaface import RetinaFace
import cv2
  
# Read the video from specified path
video = "./hdfight.mp4"
cam = cv2.VideoCapture(video) 
  
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
    
    if currentframe % 5 == 0:

        if ret:

            obj = RetinaFace.detect_faces(frame)
            len(obj.keys())

            for key in obj.keys():
                identity = obj[key]
                
                facial_area = identity["facial_area"]

                cv2.rectangle(frame, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), 1)

                

            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
  
            # writing the extracted images 
            cv2.imwrite(name, frame[:, :, ::-1]) 
  
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
