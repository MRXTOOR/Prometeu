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

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "obj.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "cfg/yolov3-spp.cfg"
modelWeights = "model/yolov3-spp.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)



# Process inputs
winName = 'Object Detection using YOLO in OPENCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)




cap = cv.VideoCapture(args.video)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



while cv.waitKey(1) < 0:
    # Get frame from the video
    hasFrame, frame = cap.read()

    if not hasFrame:
        print("Done processing!")
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Show the frame with the detection boxes
    cv.imshow(winName, frame)
    cv.waitKey(1000)  # Пауза между кадрами (в миллисекундах), можно изменить

cv.destroyAllWindows()

#------------------------------------------------------------------------------------------------

def most_frequent(List):
    occurence_count = collections.Counter(List)
    return occurence_count.most_common(1)[0][0]


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the indices of the output layers, i.e. the layers with unconnected outputs
    output_layers = net.getUnconnectedOutLayers()
    return [layersNames[i - 1] for i in output_layers]



# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)



# Remove the bounding boxes with low confidence using non-maxima suppression
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



# Process inputs
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

# Get the video writer initialized to save the output video
if os.path.isfile(video_file):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video or images

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

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, d)

    d += 1
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if os.path.isfile(video_file):
        vid_writer.write(frame.astype(np.uint8))
    else:
        cv.imwrite(os.path.join(outputFile, f"frame_{framecount}.jpg"), frame)

    if cv.waitKey(1) & 0xFF == 27:
        break