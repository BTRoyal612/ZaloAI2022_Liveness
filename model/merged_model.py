from tensorflow.keras.models import load_model
from config import face_detection, liveness_detection
import numpy as np
import cv2
import pickle
import imutils


class MergedModel:
    def __init__(self):
        self.faceNet = cv2.dnn.readNet(face_detection['path_to_proto'],
                                        face_detection['path_to_model'])
        self.livenessNet = load_model(liveness_detection['path_to_model'])
        self.livenessNet_le = pickle.loads(open(liveness_detection['path_to_le'], "rb").read())
    
    def detect_liveness(self, frame):
        # grab the dimensions of the frame and then construct a blob
	    # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                if face.any():
                    # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (32, 32))
                    # face = img_to_array(face)
                    # face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # label = le.classes_[j]
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float") / 255.0
            preds = self.livenessNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)
    
    def predict(self, video):
        cap = cv2.VideoCapture(video)

        # loop over the frames from the video stream
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = self.detect_liveness(frame)
            if len(locs) < 1: 
                continue
            loc = locs[0]
            pred = preds[0]
            
            # loop over the detected face locations and their corresponding
            # locations
            # for i in range(len(locs)):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = loc
            (fake, real) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
                
            # include the probability in the label
            label = "Liveness"
            label = "{}: {:.2f}%".format(label, real * 100)
            color = (0, 255, 0)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        
        # do a bit of cleanup
        cap.release()
        cv2.destroyAllWindows()