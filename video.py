from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.2)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3-tiny.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3-tiny_CarModel.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video1", dest = "videofile_1", help = "Video file to run detection on", default = 0, type = str)
    parser.add_argument("--video2", dest="videofile_2", help="Video file to run detection on", default=1, type=str)
    
    return parser.parse_args()

'''
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    #coordinate = (x[1:3].float() + x[3:5].float())/2

    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])# + " " + str(coordinate)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img
    '''

def write(match,Q,frame_1,frame_2):
    for i in range(len(match)):
        L1 = match[i][0][1:3]
        L2 = match[i][0][3:5]
        R1 = match[i][1][1:3]
        R2 = match[i][1][3:5]
        xl = (L1[0] + L2[0])/2
        yl = (L1[1] + L2[1])/2
        d = (xl - (R1[0] + R2[0])/2)
        xl_yl_d_1 = np.array([[xl,yl,d,1]]).transpose()
        X_Y_Z_W = np.dot(Q,xl_yl_d_1)


        #我添加
        xr = (R1[0] + R2[0]) / 2
        yr = (R1[1] + R2[1]) / 2
        xl = xl.int()
        yl = yl.int()
        xr = xr.int()
        yr = yr.int()

        SAD = ((frame_2[(yl - 20):(yl + 21),(xl - 20):(xl + 21)] - frame_1[(yr - 20):(yr + 21),(xr - 20):(xr + 21)])**2).sum()
        print(SAD)


        X =  (X_Y_Z_W[0][0] / X_Y_Z_W[3][0])-(1/9.54162454e-03)/2
        Y =  X_Y_Z_W[1][0] / X_Y_Z_W[3][0]
        Z =  X_Y_Z_W[2][0] / X_Y_Z_W[3][0]

        distance = (10.48/104.8)*(X**2 + Y**2 + Z**2)**(0.5)

        L1 = tuple(L1.int())
        L2 = tuple(L2.int())
        R1 = tuple(R1.int())
        R2 = tuple(R2.int())

        #Left
        cv2.rectangle(frame_2, L1, L2, (255,0,0), 2)
        #label = "Car" + str(i+1) + " Distance: " + str(distance) + " cm"
        label = "Car" + str(i + 1) + " Distance: %.2f cm" %distance
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = L1[0] + t_size[0] + 3, L1[1] + t_size[1] + 4
        cv2.rectangle(frame_2, L1, c2, (255,0,0), -1)
        cv2.putText(frame_2, label, (L1[0], L1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);

        #Right
        cv2.rectangle(frame_1, R1, R2, (255, 0, 0), 2)
        label = "Car" + str(i + 1) + " Distance: %.2f cm" % distance
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = R1[0] + t_size[0] + 3, R1[1] + t_size[1] + 4
        cv2.rectangle(frame_1, R1, c2, (255, 0, 0), -1)
        cv2.putText(frame_1, label, (R1[0], R1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);

    return frame_1,frame_2


def Rectify():
    cameraMatrix_L = np.array([[304.738607209809,0,305.994100364718],
                              [0,305.227789108715,241.212208314937],
                              [0,0,1]])

    cameraMatrix_R = np.array([[308.133695955619,0,315.955563318969],
                              [0,308.895190550419,235.486939104551],
                              [0,0,1]])

    distortion_L = np.array([[-0.00996972051549407,-0.0235672356917688,0,0,0]])

    distortion_R = np.array([[-0.0105265263268979,- 0.0263872924192230,0,0,0]])

    rotationMatrix = np.array([[0.999679925455426,-0.0217707405601446,0.0128872610316053],
                               [0.0220221532189026,0.999563386933189,-0.0196992454235203],
                               [-0.0124527671236974,0.0199767454335229,0.999722890721648]])

    disparityMatrix = np.array([-104.799629055824,0.671137885514781,0.675686121782042])

    size = (640, 480)


    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify( cameraMatrix_L, distortion_L,
                                                                       cameraMatrix_R, distortion_R, size, rotationMatrix,
                                                                       disparityMatrix,flags = cv2.CALIB_ZERO_DISPARITY)

    left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix_L, distortion_L, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix_R, distortion_R, R2, P2, size,cv2.CV_16SC2)

    return Q, validPixROI1, left_map1, left_map2, validPixROI2, right_map1, right_map2

    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

Q, validPixROI1, left_map1, left_map2, validPixROI2, right_map1, right_map2 = Rectify()


num_classes = 1
classes = load_classes("data/car.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



#Detection phase

videofile_1 = args.videofile_1 #or path to the video file.
videofile_2 = args.videofile_2 #or path to the video file.


cap_1 = cv2.VideoCapture(videofile_1)
assert cap_1.isOpened(), 'Cannot capture source'
cap_2 = cv2.VideoCapture(videofile_2)
assert cap_2.isOpened(), 'Cannot capture source'

#cap = cv2.VideoCapture(0)  for webcam


frames = 0  
start = time.time()

while cap_1.isOpened() & cap_2.isOpened():
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()

    frame_1 = cv2.remap(frame_1, right_map1, right_map2, cv2.INTER_LINEAR)
    frame_1 = frame_1[validPixROI2[1]:validPixROI2[1] + validPixROI2[3],
              validPixROI2[0]:validPixROI2[0] + validPixROI2[2]]

    frame_2 = cv2.remap(frame_2, left_map1, left_map2, cv2.INTER_LINEAR)
    frame_2 = frame_2[validPixROI1[1]:validPixROI1[1] + validPixROI1[3],
             validPixROI1[0]:validPixROI1[0] + validPixROI1[2]]

    if ret_1 & ret_2:
        img_1 = prep_image(frame_1, inp_dim)
        img_2 = prep_image(frame_2, inp_dim)

        im_dim_1 = frame_1.shape[1], frame_1.shape[0]
        im_dim_1 = torch.FloatTensor(im_dim_1).repeat(1,2)
        im_dim_2 = frame_2.shape[1], frame_2.shape[0]
        im_dim_2 = torch.FloatTensor(im_dim_2).repeat(1, 2)
                     
        if CUDA:
            im_dim_1 = im_dim_1.cuda()
            im_dim_2 = im_dim_2.cuda()
            img_1 = img_1.cuda()
            img_2 = img_2.cuda()
        
        with torch.no_grad():
            output_1 = model(Variable(img_1, volatile = True), CUDA)
            output_2 = model(Variable(img_2, volatile=True), CUDA)
        output_1 = write_results(output_1, confidence, num_classes, nms_conf = nms_thesh)
        output_2 = write_results(output_2, confidence, num_classes, nms_conf=nms_thesh)


        if (type(output_1) == int) | (type(output_2) == int):
            frames += 1
            #print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("Right", frame_1)
            cv2.imshow("Left", frame_2)
            key = cv2.waitKey(5)   #可改秒數!!!!!!!!!!!!!
            if key & 0xFF == ord('q'):
                break
            continue
        

        im_dim_1 = im_dim_1.repeat(output_1.size(0), 1)
        scaling_factor_1 = torch.min(416/im_dim_1,1)[0].view(-1,1)
        im_dim_2 = im_dim_2.repeat(output_2.size(0), 1)
        scaling_factor_2 = torch.min(416 / im_dim_2, 1)[0].view(-1, 1)
        
        output_1[:,[1,3]] -= (inp_dim - scaling_factor_1*im_dim_1[:,0].view(-1,1))/2
        output_1[:,[2,4]] -= (inp_dim - scaling_factor_1*im_dim_1[:,1].view(-1,1))/2
        output_1[:,1:5] /= scaling_factor_1

        output_2[:, [1, 3]] -= (inp_dim - scaling_factor_2 * im_dim_2[:, 0].view(-1, 1)) / 2
        output_2[:, [2, 4]] -= (inp_dim - scaling_factor_2 * im_dim_2[:, 1].view(-1, 1)) / 2
        output_2[:, 1:5] /= scaling_factor_2

        for i in range(output_1.shape[0]):
            output_1[i, [1,3]] = torch.clamp(output_1[i, [1,3]], 0.0, im_dim_1[i,0])
            output_1[i, [2,4]] = torch.clamp(output_1[i, [2,4]], 0.0, im_dim_1[i,1])

        for i in range(output_2.shape[0]):
            output_2[i, [1, 3]] = torch.clamp(output_2[i, [1, 3]], 0.0, im_dim_2[i, 0])
            output_2[i, [2, 4]] = torch.clamp(output_2[i, [2, 4]], 0.0, im_dim_2[i, 1])

        #u0 = 2.96011803e+02
        #v0 = 2.35611069e+02

        match = []
        for i in range(output_2.shape[0]):
            xl = ((output_2[i][1] + output_2[i][3])/2).int()
            yl = ((output_2[i][2] + output_2[i][4])/2).int()

            if output_1.shape[0] == 0:
                break

            xr = ((output_1[0][1] + output_1[0][3]) / 2).int()
            yr = ((output_1[0][2] + output_1[0][4]) / 2).int()

            SAD = ((frame_2[(yl - 20):(yl + 21),(xl - 20):(xl + 21)] - frame_1[(yr - 20):(yr + 21),(xr - 20):(xr + 21)])**2).sum()
            idx = 0
            flag = 0

            for j in range(output_1.shape[0]):
                xr = ((output_1[j][1] + output_1[j][3]) / 2).int()
                yr = ((output_1[j][2] + output_1[j][4]) / 2).int()
                if abs(xr + 104.8 - xl) > 104.8 or abs(yr - yl) > 20:
                    continue
                flag = 1
                temp = ((frame_2[(yl - 20):(yl + 21),(xl - 20):(xl + 21)] - frame_1[(yr - 20):(yr + 21),(xr - 20):(xr + 21)])**2).sum()
                if temp < SAD:
                    SAD = temp
                    idx = j
            if flag == 0:
                continue
            match.append([output_2[i],output_1[idx]])
            output_1 = np.delete(output_1, idx, axis=0)

        classes = load_classes('data/car.names')
        colors = pkl.load(open("pallete", "rb"))

        '''
        list(map(lambda x: write(x, frame_1), output_1))
        list(map(lambda x: write(x, frame_2), output_2))
        '''

        frame_1, frame_2 = write(match, Q, frame_1, frame_2)

        cv2.imshow("Right", frame_1)
        cv2.imshow("Left", frame_2)
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        #print(time.time() - start)
        #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     






