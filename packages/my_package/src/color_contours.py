#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import cv2
from cv_bridge import CvBridge

class ColorDetection(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ColorDetection, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._image_contours_topic = f"/{self._vehicle_name}/camera_node/image_contours/compressed"
        self._image_contours_publisher = rospy.Publisher(self._image_contours_topic, CompressedImage)

        self._undistorted_topic = f"/{self._vehicle_name}/camera_node/undistorted_image/compressed"
        self.sub = rospy.Subscriber(self._undistorted_topic, CompressedImage, self.callback)

        self._bridge = CvBridge()

    def callback(self, msg):
        # Convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Convert warped image to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100], dtype=np.uint8)  # Lower bound for red
        upper_red = np.array([10, 255, 255], dtype=np.uint8)  # Upper bound for red
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_blue = np.array([100, 150, 0], dtype=np.uint8)  # Lower bound for blue
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)  # Upper bound for blue
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_green = np.array([40, 40, 150], dtype=np.uint8)  # Lower bound for green (close to #56B49C)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)  # Upper bound for green (close to #56B49C)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Dilation kernel
        kernel = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(image, image, 
                                mask = red_mask) 
        
        # For green color 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(image, image, 
                                    mask = green_mask) 
        
        # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(image, image, 
                                mask = blue_mask)
        
        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2) 
                
                cv2.putText(image, "Red Colour", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (0, 0, 255))     

        # Creating contour to track green color 
        contours, hierarchy = cv2.findContours(green_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (0, 255, 0), 2) 
                
                cv2.putText(image, "Green Colour", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 255, 0)) 

        # Creating contour to track blue color 
        contours, hierarchy = cv2.findContours(blue_mask, 
                                            cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                image = cv2.rectangle(image, (x, y), 
                                        (x + w, y + h), 
                                        (255, 0, 0), 2) 
                
                cv2.putText(image, "Blue Colour", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (255, 0, 0))

        # Publish the masks
        self._image_contours_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(image))

if __name__ == '__main__':
    # create the node
    node = ColorDetection(node_name="color_detection_node")
    # keep spinning
    rospy.spin()