import cv2
import numpy as np


#cap = cv2.VideoCapture('car_chase_01.mp4')
image = cv2.imread('baggage_claim.jpg')
#print(image.shape)

whT = 320
confThreshold = 0.5
# The lower the value, the more aggressive, less boxes will be
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
#print(len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confidences = []

    for output in outputs:
        for detection in output:
            # Remove the first 5 elements
            scores = detection[5:]
            # Get the index of the maximum value
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                # Since the w and h are the percentage value, we need to multiply it by the actual size
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                # Offset the coordinate of the bbox from the center to the left top conner
                x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)

                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidences.append(float(confidence))

    #print(len(bbox))
    # Return the indices that need to keep
    indices = cv2.dnn.NMSBoxes(bbox, confidences, confThreshold, nmsThreshold)


    #print(indices)
    for i in indices:
        # The indices is in the list, so that we need to remove the bracket
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidences[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)



#success, img = cap.read()
img = image

blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
net.setInput(blob)

layerNames = net.getLayerNames()
#print(layerNames)

#print(net.getUnconnectedOutLayers())
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
#print(outputNames)

outputs = net.forward(outputNames)
#print(len(outputs))

    ##############
    # eg. (300, 85)
    # Meaning that there are 300 bonding boxes, for each box, there are 85 elements
    # The first 4 elements are center x, center y, width and height; followed by the confidence that an object is
    # presented. The rest of 80 elements are the prediction of the 80 classes
    ##############
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)

    #print(outputs[0][0])

findObjects(outputs, img)


cv2.imshow('Image', img)

cv2.waitKey(0)
