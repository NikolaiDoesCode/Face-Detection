import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

# made a cascade. cv2.data.haarcascades is base directory where all the classifiers exist. "haarcascade_frontalface_default.xml" is the name of the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
#need greyscale image
recording = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

fram_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    _, frame = cap.read()
    #gray is image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.6, 3) 
    bodies = body_cascade.detectMultiScale(gray, 1.6,5)
    # returns list of positions of faces that exist in gray. 1.3 is accuracy of the algorith
    # 5 is minimum numbers of neighbours. 5 overlapping boxes over the same face -> decides if its a face or not
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(faces) + len(bodies) > 0:
        if recording:
            timer_started=False
        else:
            recording = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, fram_size)
            print("Started recording!")
    elif recording:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                recording = False
                timer_started = False
                out.release()
                print("Recording stopped!")
        else:
            timer_started = True
            detection_stopped_time = time.time()


    if recording:
        cv2.putText(frame,"Recording",(50,50), font, 1 ,(0,0,0), 2, cv2.LINE_4)
        out.write(frame)

    

    #drawing faces
    for(x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width,y+height), (0, 0,255), 3) # (x,y) top left, bottom right, 3 is line of thickness

    for(x,y,width,height) in bodies:
        cv2.rectangle(frame, (x,y), (x+width,y+height), (0, 0,255), 3) # (x,y) top left, bottom right, 3 is line of thickness

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == ord('q'):
        print("Recording stopped!")
        break
out.release()
cap.release()
cv2.destroyAllWindows()
