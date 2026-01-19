import cv2
import datetime
import os

# Create attendance folder if not exists
if not os.path.exists("attendance"):
    os.makedirs("attendance")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        name = "User"
        date = datetime.date.today()
        time = datetime.datetime.now().strftime("%H:%M:%S")

        with open("attendance/attendance.csv", "a") as f:
            f.write(f"{name},{date},{time}\n")

    cv2.imshow("Face Recognition Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
