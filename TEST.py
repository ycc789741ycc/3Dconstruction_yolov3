# -*- coding:utf-8 -*-
import cv2
import numpy as np

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
    
def camera():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    i = 1

    while cap1.isOpened() & cap2.isOpened():
        ret1,frame1 =  cap1.read()
        ret2, frame2 = cap2.read()

        Q, validPixROI1, left_map1, left_map2, validPixROI2, right_map1, right_map2 = Rectify()
        frame1 = cv2.remap(frame1, right_map1, right_map2, cv2.INTER_LINEAR)
        frame1 = frame1[validPixROI2[1]:validPixROI2[1] + validPixROI2[3],
                  validPixROI2[0]:validPixROI2[0] + validPixROI2[2]]

        frame2 = cv2.remap(frame2, left_map1, left_map2, cv2.INTER_LINEAR)
        frame2 = frame2[validPixROI1[1]:validPixROI1[1] + validPixROI1[3],
                  validPixROI1[0]:validPixROI1[0] + validPixROI1[2]]




        cv2.imshow('frame1',frame1)
        cv2.imshow('frame2', frame2)
        k1 = cv2.waitKey(5)
        if k1 & 0xFF == ord('q'):
            break


        k2 = cv2.waitKey(1)  
        if k2 & 0xFF == ord('a'):
            cv2.imwrite(r"car_train/car_" + str(i) + ".jpg", frame1)
            i = i + 1
            print(i)
            cv2.imwrite(r"car_train/car_" +str(i)+ ".jpg",frame2)
            print(i)
            i = i + 1

camera()