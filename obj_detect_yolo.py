import numpy as np
import cv2

import time

"""
this code works on yoloV4 , yoloV4-tiny ,yoloV3 and its diff versions
for weights and config files of different yolo models refer below link
https://github.com/AlexeyAB/darknet

"""

weightsPath = r'yolov3.weights'
configPath = r'yolov3.cfg'
labelPath = r'labels.names'
imgPath = r'test4.jpeg'


def makePrediction(net,layerNames, labels, confidence,threshold):
    """ net = loaded model
    layerNames = extraced layer names from loaded model
    label = labels of coco dataset i.e. 80 classes
    confidence = confidence of prediction of model for each object
    threshold = min threshold for non maximum suppression(method to find exact bounding box for detected object)
    """
    
    #taking the height and width
    height, width,ch = image.shape
    #preprocessing the image and creating a 4 dim blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640,640), swapRB = True, crop = False)
    # start the timer 
    start = time.time()
      
    # push img through model for output
    net.setInput(blob)
    outputs = net.forward(layerNames)
    #end the timer
    end = time.time()
    print("exeution time :",end-start)
   

    boxes, confidences, classIds = extract_boxes_confidencs_classids(outputs, confidence, width, height)
    #performing non maximum supression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIds, idxs
     
def extract_boxes_confidencs_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIds = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores) 
            conf = scores[classId]

            if  conf > confidence:
                
                box = detection[0:4] * np.array([width, height, width, height])
                cx, cy, w, h = box.astype('int')

                x = int(cx - (w/2))
                y = int(cy-(h/2))

                boxes.append([x,y,int(w),int(h)])
                confidences.append(float(conf))
                classIds.append(classId)
    
    return boxes, confidences, classIds

def draw_bounding_box(image, boxes, confidences, classIds, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classIds[i]]]
            cv2.rectangle(image, (x,y),(x+w,y+h), color, 2)
            text = "{} : {:.4f}".format(labels[classIds[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image, x,y,w,h
#storing labels in a list
file = open(labelPath,'r')
labels = file.read().strip().split('\n')


colors = np.random.randint(0,255,size = (len(labels),3),dtype = 'uint8')


#loading the model
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
#extracting names of all layers
layerNames = net.getLayerNames()
#finding the unconnected layers ie. yolo layers
layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
'''cap = cv2.VideoCapture(r'/home/jay/Pictures/internship_proj_jay/yolo/testv1.webm')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    image = cv2.resize(frame, (512,512))
    

    boxes, confidences, classIds, idxs = makePrediction(net, layerNames, labels, confidence = 0.30,threshold = 0.5)
    image  = draw_bounding_box(image, boxes, confidences, classIds, idxs, colors)
 
    # Display the resulting frame
    cv2.imshow('Frame',image)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
'''
frame = cv2.imread(r"D:\DogBreedClassification\OIP (2).jpg")
image = cv2.resize(frame, (512,512))
    

boxes, confidences, classIds, idxs = makePrediction(net, layerNames, labels, confidence = 0.30,threshold = 0.5)
image  = draw_bounding_box(image, boxes, confidences, classIds, idxs, colors)
 
    # Display the resulting frame
cv2.imshow('Frame',image)
    
    # Press Q on keyboard to  exit
cv2.waitKey(0)
cv2.destroyAllWindows()

