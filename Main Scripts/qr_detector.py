#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import Int32MultiArray, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from qrdet import QRDetector
import datetime
import pickle
#from tensorflow.keras.models import load_model
from keras.models import load_model
from collections import deque
from statistics import mode

class Image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/tello/image_raw",
                                          Image,
                                          self.callback,
                                          queue_size=1,
                                          buff_size=2**24)
        self.pubCenterQR = rospy.Publisher('/qr_detection/center',
                                           Int32MultiArray,
                                           queue_size= 10)
        self.pubQuadPoints = rospy.Publisher('/qr_detection/quad_points',
                                             Int32MultiArray,
                                             queue_size= 10)
        
        self.pubPredictedHeight = rospy.Publisher('/qr_detection/predicted_height',
                                                  Int32,
                                                  queue_size=10)
        
        # QR code detector - YOLO V8 based
        self.qr_detector = QRDetector(model_size='s')
        self.center_detections = Int32MultiArray()
        self.quad_points_msg = Int32MultiArray()

        # Height estimator - MLP scaler and model
        with open('/home/daniel/catkin_ws/src/qr_detector_tello/src/scaler_mlp_qr10.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # Mapper for category to height conversion
        with open('/home/daniel/catkin_ws/src/qr_detector_tello/src/index_to_label_qr10.pkl', 'rb') as f:
            self.index_to_label = pickle.load(f)
        
        # Load MLP classifier
        self.classifier_model = load_model('/home/daniel/catkin_ws/src/qr_detector_tello/src/qr_height_mlp_qr10.h5')
        
        # Initialize list to store data window
        self.input_features_list = []
        self.w_count = 0


    def qr_detection(self, image):
        detections = self.qr_detector.detect(image=image, is_bgr=True)
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection['bbox_xyxy'])
                quad_points = np.array(detection['quad_xy'], np.int32)
                quad_points = quad_points.reshape((-1, 1, 2))
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.polylines(image, [quad_points], isClosed=True, color=(255, 0, 0), thickness=2)  # Azul
                xmid = int((x1 + x2) / 2)
                ymid = int((y1 + y2) / 2)
                cv2.circle(image, (xmid, ymid), radius=3, color=(0, 0, 255), thickness=3)
                data = f'Center X: {xmid}, Y: {ymid}'
                #cv2.putText(image, data, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                rospy.loginfo(data)
                
                # Publicar center_detections
                self.center_detections.data = [xmid, ymid]
                self.pubCenterQR.publish(self.center_detections)

                # QR bounding box area (in pixels) and QR real size in cm
                bb_area = abs(x2 - x1) * abs(y2 - y1)
                qr_size = 10
                height = 5

                # Crear el array de características para la predicción
                #input_features = np.array([[qr_size, bb_area, xmid, ymid]])
                self.input_features_list.append([qr_size, bb_area, xmid, ymid])
                self.w_count += 1

                # If window is complete:
                if (self.w_count == 4):

                    # Reset counter
                    self.w_count = 0 

                    # Convertir la ventana a un array numpy
                    input_features = np.array(self.input_features_list)
                    
                    # Escalar las características
                    input_features_scaled = self.scaler.transform(input_features)
                    
                    # Hacer predicciones con el modelo
                    preds = self.classifier_model.predict(input_features_scaled)
                    pred_classes = np.argmax(preds, axis=1)
                    
                    # Obtener la predicción que más se repite
                    mode_class = mode(pred_classes)
                    pred_height = int(self.index_to_label[mode_class])

                    # Publicar predicted_height
                    rospy.loginfo(f'Most common height pred: {pred_height}')
                    self.pubPredictedHeight.publish(pred_height) 
                    self.input_features_list = []
                    input_features = []
                
                
                # Publicar las coordenadas de los puntos del cuadrilátero
                #xx1, yy1 = detection['quad_xy'][0]
                #xx2, yy2 = detection['quad_xy'][1]
                x3, y3 = detection['quad_xy'][2]
                x4, y4 = detection['quad_xy'][3]
                self.quad_points_msg.data = [int(x3), int(y3), int(x4), int(y4)]  # Convertir a enteros
                self.pubQuadPoints.publish(self.quad_points_msg)

                ## Crear una cadena de texto con los datos y el timestamp
                #timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #qr_points = f"{timestamp}, {qr_size}, {x1}, {y1}, {x2}, {y2}, {bb_area}, {xmid}, {ymid}, {int(xx1)}, {int(yy1)}, {int(xx2)}, {int(yy2)}, {int(x3)}, {int(y3)}, {int(x4)}, {int(y4)}, {height}"     
                
                ## Guardar los datos en un archivo de texto
                #f_name = "qr_points_" + str(height) + ".txt"
                #with open(f_name, "a") as file:  # Abrir el archivo en modo de añadir
                #    file.write(qr_points + "\n")


        return image
    
    

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")                         
        except CvBridgeError as e:
            print(e)
        #cv_image = self.qr_detection(cv_image)
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)

        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Llamar a qr_detection con la imagen en escala de grises
        gray_image_with_detection = self.qr_detection(gray_image)
        
        # Mostrar la imagen con detección
        cv2.imshow("Image window", gray_image_with_detection)
        cv2.waitKey(3)
    

def main(args):
    ic = Image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
