#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8, Int32
from std_msgs.msg import Empty
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

class FollowFaceTello:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_cmd_vel = rospy.Publisher('/tello/cmd_vel', Twist, queue_size=10)
        self.pub_land = rospy.Publisher('/tello/land', Empty, queue_size=10)
        self.Ovr = rospy.Subscriber('/keyboard/override', Int8, self.callBackFlag)
        self.detMsg = rospy.Subscriber('/qr_detection/center', 
                                        Int32MultiArray,
                                        self.detCallBack,
                                        queue_size=100)
        self.quadMsg = rospy.Subscriber('/qr_detection/quad_points', 
                                        Int32MultiArray,
                                        self.quadCallBack,
                                        queue_size=100)
        self.predictedHeightSub = rospy.Subscriber('/qr_detection/predicted_height',
                                                   Int32,
                                                   self.heightCallBack,
                                                   queue_size=100)
        self.image_sub = rospy.Subscriber('/tello/image_raw',
                                          Image,
                                          self.callback,
                                          queue_size=1, 
                                          buff_size=2**24) 
        self.vel_msg = Twist()
        self.speed_values = 0.2
        self.run = 0
        self.image_h = int(rospy.get_param("Camera.height"))
        self.image_w = int(rospy.get_param("Camera.width"))
        self.image_reduction = rospy.get_param("/followQRTello/imageReduction")
        self.image_h = int(self.image_h * (self.image_reduction/100))
        self.image_w = int(self.image_w * (self.image_reduction/100))
        self.height_by_2 = int(self.image_h/2)
        self.width_by_2 = int(self.image_w/2)
        self.centerXY = [self.width_by_2, self.height_by_2]
        self.quadPoints = []
        self.predicted_height = 200

    def callBackFlag(self, msg):
        rospy.loginfo(f'Override: {msg.data}')
        if msg.data == 5:
            self.run = 1
            rospy.loginfo("Autonomus Mode ON")
        else:
            self.run = 0
            rospy.loginfo("Autonomus Mode OFF")

    def detCallBack(self, msg):
        self.centerXY = msg.data

    def quadCallBack(self, msg):
        self.quadPoints = msg.data
      
    def heightCallBack(self, msg):
        self.predicted_height = msg.data

    def control_flight(self):
        # Calcular error en plano (x,y) de la cámara
        qr_cx = self.centerXY[0]
        qr_cy = self.centerXY[1]
        # Error entre la posición del QR y la posición deseada
        error_x = qr_cx - self.width_by_2 
        error_y = qr_cy - self.height_by_2  
        # Calcular error en z (altura del dron)
        mav_h = self.predicted_height
        error_z = 0 - mav_h
        rospy.loginfo(f'error x: {error_x}, error_y: {error_y}, error_z: {error_z}')

        # Calcular el error angular en z (ángulo entre los vectores qr y horizontal)
        if self.quadPoints:
            x3, y3, x4, y4 = self.quadPoints
            vector1 = np.array([x3 - x4, y3 - y4])
            vector2 = np.array([x4, 0])  # Vector horizontal
            error_yaw = self.angle_between_vectors(vector1, vector2)
            if(y4 > y3): error_yaw = -error_yaw
            rospy.loginfo(f'error yaw: {error_yaw}')
        
            if self.run == 1:
                if (abs(error_x) < 50 and abs(error_y) < 50 and abs(error_yaw) < 10):
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    self.vel_msg.linear.z = 0.0
                    self.vel_msg.angular.z = 0.0
                    self.pub_cmd_vel.publish(self.vel_msg)
                    rospy.loginfo(f'QR marker centered')
                    if(abs(error_z) <= 30):
                        rospy.loginfo(f'Landing')
                        self.pub_land.publish(Empty())
                        time.sleep(5)
                    else:
                        control_signal_z = 0.25 * (error_z) / 100
                        self.vel_msg.linear.z = round(control_signal_z, 2)
                        self.pub_cmd_vel.publish(self.vel_msg)
                        rospy.loginfo(f'Centered, descend')
                        
                else:
                    # Fix sign (direction) of control signals for x and y using k
                    kx = -0.24 if error_x >= 0 else 0.24
                    ky = 0.24 if error_y >= 0 else -0.24
                    # Señal de control calculada por el controlador proporcional
                    control_signal_yaw = 0.4 * error_yaw / 180
                    rospy.loginfo(f'signal yaw l(+) r(-) {control_signal_yaw}')
                    control_signal_y = kx * abs(error_x) / self.width_by_2
                    control_signal_x = ky * abs(error_y) / self.height_by_2
                    rospy.loginfo(f'signal forward(+) back(-) {control_signal_x}')
                    rospy.loginfo(f'signal left(+) right(-) {control_signal_y}') 
                    
                    if(abs(error_z) <= 30):
                        control_signal_z = 0.0
                    else:
                        control_signal_z = 0.25 * (error_z) / 100
                        rospy.loginfo(f'signal desc (-) {control_signal_z}')
            

                    self.vel_msg.linear.x = round(control_signal_x, 2)
                    self.vel_msg.linear.y = round(control_signal_y, 2)
                    self.vel_msg.linear.z = round(control_signal_z, 2)
                    self.vel_msg.angular.z = round(control_signal_yaw, 2)
                    self.pub_cmd_vel.publish(self.vel_msg)
        else:
            rospy.logwarn("No quad points available")
            error_yaw = 0  # Si no hay puntos del cuadrilátero, establece el error de yaw a 0
            self.vel_msg.linear.x = 0
            self.vel_msg.linear.y = 0
            self.vel_msg.linear.z = 0
            self.vel_msg.angular.z = 0
            self.pub_cmd_vel.publish(self.vel_msg)


                

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.control_flight()


    def angle_between_vectors(self, v1, v2):
        # Normalización de los vectores
        v1_normalized = v1 / np.linalg.norm(v1)
        v2_normalized = v2 / np.linalg.norm(v2)
        
        # Cálculo del producto punto
        dot_product = np.dot(v1_normalized, v2_normalized)
        
        # Cálculo del ángulo en radianes
        angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convertir de radianes a grados
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

def main():
    rospy.init_node('followQRTello', anonymous=True)
    followFaceTello = FollowFaceTello()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

    
