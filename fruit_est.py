# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
import random
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

class FruitPosition:

    def __init__(self, file_output):

        self.image_file = file_output
        self.camera_matrix = self.get_camera_matrix()

    # Get camera matrix
    def get_camera_matrix(self):
        # camera_matrix = np.ones((3,3))/2
        fileK = "{}intrinsic.txt".format('./calibration/param/')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        return camera_matrix

    # Get target map
    def get_target_info(self):
        # estimate pose of targets in each detector output
        target_map = {}      

        completed_img_dict = self.get_image_info(self.image_file)
        target_map = self.get_distance_angle(completed_img_dict)

        return target_map

    # use the machinevision toolbox to get the bounding box of the detected target(s) in an image
    def get_bounding_box(self, target_number, image_array):
        image = PIL.Image.fromarray(image_array.astype('uint8')).resize((640,480))
        target = Image(image)==target_number
        blobs = target.blobs()

        # Get larger blob only for the same fruit
        larger_blob_index = 0
        larger_blob_area = 0

        for i in range(len(blobs)):

            if blobs[i].area > larger_blob_area:
                larger_blob_area = blobs[i].area
                larger_blob_index = i

        [[u1,u2],[v1,v2]] = blobs[larger_blob_index].bbox # bounding box
        width = abs(u1-u2)
        height = abs(v1-v2)
        center = np.array(blobs[larger_blob_index].centroid).reshape(2,)
        box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
        # plt.imshow(fruit.image)
        # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
        # plt.show()
        # assert len(blobs) == 1, "An image should contain only one object of each target type"
        return box

    # read in the list of detection results with bounding boxes and their matching robot pose info
    def get_image_info(self, image):

        # there are at most five types of targets in each image
        target_lst_box = [[], [], [], [], []]
        completed_img_dict = {}

        # add the bounding box info of each target in each image
        # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
        image = Image(image).image
        img_vals = set(image.reshape(-1))
        for target_num in img_vals:
            if target_num > 0:
                try:
                    box = self.get_bounding_box(target_num, image) # [x,y,width,height]
                    target_lst_box[target_num-1].append(box) # bouncing box of target
                except ZeroDivisionError:
                    pass

        # if there are more than one objects of the same type, combine them
        for i in range(5):
            if len(target_lst_box[i])>0:
                box = np.stack(target_lst_box[i], axis=1)
                completed_img_dict[i+1] = {'target': box}
            
        return completed_img_dict

    def get_distance_angle(self, completed_img_dict):
        camera_matrix = self.camera_matrix
        focal_length = camera_matrix[0][0]
        # actual sizes of targets [For the simulation models]
        # You need to replace these values for the real world objects
        target_dimensions = []
        redapple_dimensions = [0.074, 0.074, 0.099]
        target_dimensions.append(redapple_dimensions)
        greenapple_dimensions = [0.081, 0.081, 0.080]
        target_dimensions.append(greenapple_dimensions)
        orange_dimensions = [0.075, 0.075, 0.075]
        target_dimensions.append(orange_dimensions)
        mango_dimensions = [0.113, 0.067, 0.062] # measurements when laying down
        target_dimensions.append(mango_dimensions)
        capsicum_dimensions = [0.073, 0.073, 0.099]
        target_dimensions.append(capsicum_dimensions)

        target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

        target_pose_dict = {}
        # for each target in each detection output, estimate its pose
        for target_num in completed_img_dict.keys():
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            true_height = target_dimensions[target_num-1][2]
            
            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            # This is the default code which estimates every pose to be (0,0)
            # target_pose = {'x': 0.0, 'y': 0.0}
            target_info = {'distance':0.0, 'angle':0.0}

            # Distance from camera to fruit
            distance_to_fruit = (true_height * focal_length) / (box[3])

            # Angle from the middle of frame to fruit
            x_middle_to_fruit = box[0] - 320
            angle_from_center = np.arctan(x_middle_to_fruit/focal_length)

            target_info = {'distance':float(distance_to_fruit), 'angle':float(angle_from_center)}

            target_pose_dict[target_list[target_num-1]] = target_info
            ###########################################
        
        return target_pose_dict
