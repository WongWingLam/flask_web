from flask import Flask, render_template
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import urllib
import mysql.connector
import pyttsx3
from datetime import datetime
import sys

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/faceCapture/')
def faceCapture():
    faceCascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    # Specify the `user_name` and `NUM_IMGS` here.
    user_name = "Jacky"
    NUM_IMGS = 50
    if not os.path.exists('data/{}'.format(user_name)):
        os.mkdir('data/{}'.format(user_name))

    cnt = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (80, 50)
    fontScale = 1
    fontColor = (102, 102, 225)
    lineType = 2

    # Open camera
    while cnt <= NUM_IMGS:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        # Store the captured images in `data/Jack`
        cv2.imwrite("data/{}/{}{:03d}.jpg".format(user_name, user_name, cnt), frame)
        cnt += 1

        key = cv2.waitKey(100)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    return render_template('index.html')

@app.route('/train/')
def train():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  image_dir = os.path.join(BASE_DIR, "data")

  # Load the OpenCV face recognition detector Haar
  face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
  # Create OpenCV LBPH recognizer for training
  recognizer = cv2.face.LBPHFaceRecognizer_create()

  current_id = 0
  label_ids = {}
  y_label = []
  x_train = []

  # Traverse all face images in `data` folder
  for root, dirs, files in os.walk(image_dir):
      for file in files:
          if file.endswith("png") or file.endswith("jpg"):
              path = os.path.join(root, file)
              label = os.path.basename(root).replace("", "").upper()  # name
              print(label, path)

              if label in label_ids:
                  pass
              else:
                  label_ids[label] = current_id
                  current_id += 1
              id_ = label_ids[label]
              print(label_ids)

              pil_image = Image.open(path).convert("L")
              image_array = np.array(pil_image, "uint8")
              print(image_array)
              # Using multiscle detection
              faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)

              for (x, y, w, h) in faces:
                  roi = image_array[y:y+h, x:x+w]
                  x_train.append(roi)
                  y_label.append(id_)

  # labels.pickle store the dict of labels.
  # {name: id}  
  # id starts from 0
  with open("labels.pickle", "wb") as f:
      pickle.dump(label_ids, f)

  # Train the recognizer and save the trained model.
  recognizer.train(x_train, np.array(y_label))
  recognizer.save("train.yml")
  return render_template('index.html')

@app.route('/face/')
def face():
  # 1 Create database connection
  myconn = mysql.connector.connect(host="localhost", user="root", passwd="1234", database="facerecognition")
  date = datetime.utcnow()
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  cursor = myconn.cursor()


  #2 Load recognize and read label from model
  recognizer = cv2.face.LBPHFaceRecognizer_create()
  recognizer.read("train.yml")

  labels = {"person_name": 1}
  with open("labels.pickle", "rb") as f:
      labels = pickle.load(f)
      labels = {v: k for k, v in labels.items()}

  # create text to speech
  engine = pyttsx3.init()
  rate = engine.getProperty("rate")
  engine.setProperty("rate", 175)

  # Define camera and detect face
  face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
  cap = cv2.VideoCapture(0)

  # 3 Open the camera and start face recognition
  while True:
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

      for (x, y, w, h) in faces:
          print(x, w, y, h)
          roi_gray = gray[y:y + h, x:x + w]
          roi_color = frame[y:y + h, x:x + w]
          # predict the id and confidence for faces
          id_, conf = recognizer.predict(roi_gray)

          # 3.1 If the face is recognized
          if conf >= 60:
              # print(id_)
              # print(labels[id_])
              font = cv2.QT_FONT_NORMAL
              id = 0
              id += 1
              name = labels[id_]
              current_name = name
              color = (255, 0, 0)
              stroke = 2
              cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
              cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), (2))

              # Find the customer's information in the database.
              select = "SELECT customer_id, name, DAY(login_date), MONTH(login_date), YEAR(login_date) FROM Customer WHERE name='%s'" % (name)
              name = cursor.execute(select)
              result = cursor.fetchall()
              # print(result)
              data = "error"

              for x in result:
                  data = x

              # If the customer's information is not found in the database
              if data == "error":
                  print("The customer", current_name, "is NOT FOUND in the database.")

              # If the customer's information is found in the database
              else:
                  """
                  Implement useful functions here.
                  

                  """
                  # Update the data in database
                  update =  "UPDATE Customer SET login_date=%s WHERE name=%s"
                  val = (date, current_name)
                  cursor.execute(update, val)
                  update = "UPDATE Customer SET login_time=%s WHERE name=%s"
                  val = (current_time, current_name)
                  cursor.execute(update, val)
                  myconn.commit()
                
                  hello = ("Hello ", current_name, "Welcom to the iKYC System")
                  print(hello)
                  engine.say(hello)
                  # engine.runAndWait()


          # 3.2 If the face is unrecognized
          else: 
              color = (255, 0, 0)
              stroke = 2
              font = cv2.QT_FONT_NORMAL
              cv2.putText(frame, "UNKNOWN", (x, y), font, 1, color, stroke, cv2.LINE_AA)
              cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), (2))
              hello = ("Your face is not recognized")
              print(hello)
              engine.say(hello)
              # engine.runAndWait()

      cv2.imshow('iKYC System', frame)
      k = cv2.waitKey(20) & 0xff
      if k == ord('q'):
          break
          
  cap.release()
  cv2.destroyAllWindows()
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)