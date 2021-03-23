import cv2
import dlib
from scipy.spatial import distance
import openpyxl
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import ResultsToDataFrame

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("cascades/shape_predictor_68_face_landmarks.dat")


def mouth_detection(image):
    img = image

    face_mask = 0
    if img.size != 0:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mouths = mouth_cascade.detectMultiScale(gray, 1.1, 4)


        for mouth in mouths:

            if mouths.size != 0:
                face_mask = 0

            else:
                face_mask = 1

    return face_mask


def gray_scaling(image, threshold):
    face_mask = 0

    if image.size != 0:

        img = cv2.resize(image, (960, 540))

        lower = np.array([89, 47, 42])
        upper = np.array([197, 140, 133])
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        total_number_of_pixels = output.shape[1] * output.shape[0]
        number_of_not_black_pix = np.sum(output != 0)
        number_of_black_pix = total_number_of_pixels - number_of_not_black_pix

        percentage_of_black_pixel = (number_of_black_pix / total_number_of_pixels) * 100

        if percentage_of_black_pixel >= threshold:
            face_mask = 1
        else:
            face_mask = 0

    return face_mask


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

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

            if face.size != 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locs, preds


def image_classification(roi_color):
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("cascades/mask_detector.model")
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    face_mask = 0

    frame = roi_color
    if frame.size != 0:

        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            (mask, withoutMask) = pred

            # include the probability in the label
            if mask > withoutMask:
                face_mask = 1
            else:
                face_mask = 0

    # Return decision
    return face_mask