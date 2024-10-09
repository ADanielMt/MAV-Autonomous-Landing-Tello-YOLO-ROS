# YOLOv8 based autonomous landing system for Tello MAV 


In this repository, you can find the step by step, scripts and demonstrative videos of an 
autonomous landing system for a Micro Aerial Vehicle, also know ans micro drones. 
In this master thesis project: 
 * You Only Look Once version 8 (YOLOv8) detector is used to identify a QR landing marker. 
 * A Multi-Layer Perceptron (MLP) is trained to estimate the relative height between the 
MAV and the moving platform. 
 * A Proportional-Integral (PI) controller calculates and sends control commands to the MAV 
for landing, using Robot Operating System (ROS) as an interface. 
 * The landing is performed
on a mobile platform.

## Environment Setup

To be added. In the meantime, follow the txt guides in Environment setup folder.

## Integrated system test

In order to run the integrated system with the Tello drone, the following commands and 
scripts must be executed on individual linux terminals:

*  ``` roscore   ## Execute master node ```
* ``` rosrun driver_tello_mod tello_driver_mod.py     ## Execute tello driver node ```
* ``` rosrun keyboard tello_keyboard.py  ## Start manual (keyboard) control node ```
* ``` rosrun qr_detector_tello qr_detector.py  ## Start qr_detector node ```
* ``` roslaunch follow_qr_tello follow_qr_tello.launch image_reduction:=60  ## Start tello_lander node ```


Using the manual mode of the manual controller module (tello_keyboard.py), 
take-off the drone and locate it over the QR landing marker. When a portion of the 
QR is visible in the camera, activate the autonomous mode with the keyboard control.
The autonomous lander will start working, tracking the QR and landing over it.



<img src="/Assets/Integrated_system.jpg" width="500">

## To be done...

* Add step-by-step tutorial in README file
* Add requirements.txt (Currently, you can find the required packages in Environment setup files)
* Add ROS workspace files
* Add drivers used (Currently, you can find the download links in Environment setup files)
* Add videos of the tests performed (Tello camera and ROS/linux terminals video recording)