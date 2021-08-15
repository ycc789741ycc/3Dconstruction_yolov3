# 3Dconstruction_yolov3 
This is a project about using two cameras to detect car model's distance from the carmera   
position, and this project applied yolov3 to capture the same target on each camera screen.  
(If you want to use this sample code, you may need train your yolov3 model and tune  
camera's matrix using matlab due to different physical condition.)

## Environment
Python 3.7.2  
torch==1.2.0+cpu  
torchvision==0.4.0+cpu  
opencv-python==4.1.0.25  
numpy==1.17.1  
pandas==0.25.1  

## Hardware requirement  
Dual lens camera

## Demo  
This is the screenshoot from one of two camera.  
![Demo Pic](https://github.com/ycc789741ycc/3Dconstruction_yolov3/blob/master/Demo.png "Demo Pic")

## Quick Start 
cd C:\File's Path\Project_Yolov3_Distance_Detection  
python video.py  

## Detail 
[專題_基於YOLOv3執行雙目測距](https://github.com/ycc789741ycc/3Dconstruction_yolov3/blob/master/%E5%B0%88%E9%A1%8C_%E5%9F%BA%E6%96%BCYOLOv3%E5%9F%B7%E8%A1%8C%E9%9B%99%E7%9B%AE%E6%B8%AC%E8%B7%9D.pdf)  
