import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import os
from datetime import datetime
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import collections

font = cv.FONT_HERSHEY_SIMPLEX

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

classesFile = "obj.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "cfg/yolov3-spp.cfg"
modelWeights = "model/yolov3-spp.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)



winName = 'Object Detection using YOLO in OPENCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)




cap = cv.VideoCapture(args.video)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Done processing!")
        cv.waitKey(3000)
        break

    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

  
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv.imshow(winName, frame)
    cv.waitKey(1000) 

cv.destroyAllWindows()

#------------------------------------------------------------------------------------------------

def most_frequent(List):
    occurence_count = collections.Counter(List)
    return occurence_count.most_common(1)[0][0]


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    return [layersNames[i - 1] for i in output_layers]



def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)



def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                drawPred(classId, confidence, left, top, left + width, top + height)



winName = 'Resource Detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "resource_detection_output.avi"
video_file = "video.mp4"
image_folder = "Labelled"

if os.path.isfile(video_file):
    cap = cv.VideoCapture(video_file)
    outputFile = video_file[:-4] + '_resource_detection_output.avi'
elif os.path.isdir(image_folder):
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images.sort()
    outputFile = image_folder + '_resource_detection_output.avi'
else:
    print("No valid input file or folder found.")
    sys.exit(1)

if os.path.isfile(video_file):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:


    if os.path.isfile(video_file):
        hasFrame, frame = cap.read()
    else:
        if framecount < len(images):
            frame = cv.imread(images[framecount])
            hasFrame = True
        else:
            hasFrame = False

    framecount += 1

    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs, d)

    d += 1

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

 
    if os.path.isfile(video_file):
        vid_writer.write(frame.astype(np.uint8))
    else:
        cv.imwrite(os.path.join(outputFile, f"frame_{framecount}.jpg"), frame)

    if cv.waitKey(1) & 0xFF == 27:
        break