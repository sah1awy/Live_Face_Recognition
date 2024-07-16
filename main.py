import threading 
import cv2 as cv
from deepface import DeepFace


cap = cv.VideoCapture(0,cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)

counter = 0

face_match = False

reference_img = cv.imread("reference.jpg")


def check_face(frame):
    global face_match 
    try:
        if DeepFace.verify(frame,reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False

    except ValueError:
        face_match = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face,args=(frame.copy(),)).start()

            except ValueError:
                pass
        counter += 1

        if face_match:
            cv.putText(frame,"MATCH!",(20,450),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3) 
        else:   
            cv.putText(frame,"NO MATCH!",(20,450),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3) 

        cv.imshow("video",frame)


    key = cv.waitKey(1)

    if key == ord("q"):
        break


cv.destroyAllWindows()
