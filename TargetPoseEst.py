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


# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
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
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    print(img_vals)
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
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
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        # This is the default code which estimates every pose to be (0,0)
        target_pose = {'x': 0.0, 'y': 0.0}

        # Distance from camera to fruit
        distance_to_fruit = (true_height * focal_length) / (box[3])

        # Angle from the middle of frame to fruit
        x_middle_to_fruit = box[0] - 320
        angle_from_center = np.arctan(x_middle_to_fruit/focal_length)

        # Fruit position in world frame
        angle_world_to_fruit = - robot_pose[2] + angle_from_center
        fruit_world_x = robot_pose[0] + (distance_to_fruit * np.cos(angle_world_to_fruit)) 
        fruit_world_y = robot_pose[1] - (distance_to_fruit * np.sin(angle_world_to_fruit)) 
        target_pose = {'x': float(fruit_world_x), 'y': float(fruit_world_y)}

        target_pose_dict[target_list[target_num-1]] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 1 estimate for each target type
def merge_estimations(target_map):
    target_map = target_map
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    target_est = {}
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('redapple'):
                redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
    # Replace it with a better merge solution.

    def mean(input):
        mean = np.mean(input)
        # accuracy = 0.4
        # fil = round(accuracy * round(mean/accuracy),4)
        return mean

    if len(redapple_est) >= num_per_target:
        redapple_x = [pose[0] for pose in redapple_est]
        redapple_y = [pose[1] for pose in redapple_est]
        avg_redapple_x = mean(redapple_x)
        avg_redapple_y = mean(redapple_y)

    if len(greenapple_est) >= num_per_target:
        greenapple_x = [pose[0] for pose in greenapple_est]
        greenapple_y = [pose[1] for pose in greenapple_est]
        avg_greenapple_x = mean(greenapple_x)
        avg_greenapple_y = mean(greenapple_y)

    if len(orange_est) >= num_per_target:
        orange_x = [pose[0] for pose in orange_est]
        orange_y = [pose[1] for pose in orange_est]
        avg_orange_x = mean(orange_x)
        avg_orange_y = mean(orange_y)

    if len(mango_est) >= num_per_target:
        mango_x = [pose[0] for pose in mango_est]
        mango_y = [pose[1] for pose in mango_est]
        avg_mango_x = mean(mango_x)
        avg_mango_y = mean(mango_y)
    
    if len(capsicum_est) >= num_per_target:
        capsicum_x = [pose[0] for pose in capsicum_est]
        capsicum_y = [pose[1] for pose in capsicum_est]
        avg_capsicum_x = mean(capsicum_x)
        avg_capsicum_y = mean(capsicum_y)
        

    for i in range(num_per_target):
        try:
            target_est['redapple_'+str(i)] = {'x':avg_redapple_x, 'y':avg_redapple_y}
        except:
            pass
        try:
            target_est['greenapple_'+str(i)] = {'x':avg_greenapple_x, 'y':avg_greenapple_y}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'x':avg_orange_x, 'y':avg_orange_y}
        except:
            pass
        try:
            target_est['mango_'+str(i)] = {'x':avg_mango_x, 'y':avg_mango_y}
        except:
            pass
        try:
            target_est['capsicum_'+str(i)] = {'x':avg_capsicum_x, 'y':avg_capsicum_y}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are only one estimate for each target type
    target_est = merge_estimations(target_map)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')