import cv2
import numpy as np
from keras.preprocessing import image
import time
import argparse
import imutils
import pickle
import os


def nothing(x):
    pass
'''
image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
    test_image = image.load_img('1.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    
    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'
    else:
        return "None"
'''
def recognizeHand(path):
    print(path)
    #python recognizeface.py --detector face_detection_model
    #--embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image images/adrian.jpg

    text=""

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detection_model",
            "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    print("Captured")

    # loop over the detections
    for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                            continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                            (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # perform classification to recognize the face
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]

                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    text=name
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    print("hi")

    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    cv2.imwrite("1.png", image)
    time.sleep(1)
    #text = predictor()
    connection=mysql.connector.connect(host='localhost',database='policebeatsdb',user='root',password='')
    cursor = connection.cursor()
    sq_query="select pid,station,bnum from p21pb_police where pname='"+text+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    pid = data[0][0]
    stat=data[0][1]
    bnum=data[0][2]
    
    connection.commit() 
    connection.close()
    cursor.close()
    print(text)
    val=text+","+pid+","+stat+","+bnum
    return val
def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    s = measure.compare_ssim(imageA, imageB, multichannel=True)
    return s

def recognizeImage(path):
    print(path)
    text=""
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Conv2D
    Classifier=Sequential()

    Classifier.add(Conv2D(32,(3,3), input_shape=(224,224,3), activation='relu'))
    Classifier.add(MaxPooling2D(pool_size=(2,2)))

    Classifier.add(Conv2D(32,(3,3),activation='relu'))
    Classifier.add(MaxPooling2D(pool_size=(2,2)))

    Classifier.add(Flatten())

    Classifier.add(Dense(units = 128, activation = 'relu'))
    Classifier.add(Dense(units = 7, activation = 'softmax'))

    Classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'] )
    
    model=load_model('newmodel_Classifier.h5')
    img=image.load_img(path,target_size=(224,224))
    test_image=image.img_to_array(img)
    test_image=np.expand_dims(test_image, axis = 0)
    result = Classifier.predict(test_image)
    print(result)
    a=np.argmax(model.predict(test_image), axis=1)
    print(a[0])
    return str(a[0])

recognizeImage("a.jpeg")
#predictor()
