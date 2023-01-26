# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image
from Astar import AStarPlanner, Square

import matplotlib.pyplot as plt
import PIL
import time

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
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
    redapple_dimensions = [0.074, 0.074, 0.087]
    target_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067]
    target_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072]
    target_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
    target_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.088]
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

        # Distance from Robot Camera to Fruit
        distance_to_fruit = (true_height * focal_length) / box[3]

        # Angle from the Middle of Frame to Fruit relative to x-axis
        frame_width = 320
        middle_to_fruit = box[0] - frame_width
        x_angle_from_center = np.arctan(middle_to_fruit/focal_length)

        # Fruit position in World Frame (wf)
        angle_wf_to_fruit = x_angle_from_center - robot_pose[2]
        fworld_x = robot_pose[0] + (distance_to_fruit * np.cos(angle_wf_to_fruit)) 
        fworld_y = robot_pose[1] - (distance_to_fruit * np.sin(angle_wf_to_fruit)) 
        target_pose = {'x': float(fworld_x), 'y': float(fworld_y)}

        target_pose_dict[target_list[target_num-1]] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 1 estimate for each target type
def merge_estimations(target_map):
    target_map = target_map
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
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
    if len(redapple_est) >= num_per_target:
        redapple_x = [pose[0] for pose in redapple_est]
        redapple_xfinal = np.mean(redapple_x)
        redapple_y = [pose[1] for pose in redapple_est]
        redapple_yfinal = np.mean(redapple_y)
    if len(greenapple_est) >= num_per_target:
        greenapple_x = [pose[0] for pose in greenapple_est]
        greenapple_xfinal = np.mean(greenapple_x)
        greenapple_y = [pose[1] for pose in greenapple_est]
        greenapple_yfinal = np.mean(greenapple_y)
    if len(orange_est) >= num_per_target:
        orange_x = [pose[0] for pose in orange_est]
        orange_xfinal = np.mean(orange_x)
        orange_y = [pose[1] for pose in orange_est]
        orange_yfinal = np.mean(orange_y)
    if len(mango_est) >= num_per_target:
        mango_x = [pose[0] for pose in mango_est]
        mango_xfinal = np.mean(mango_x)
        mango_y = [pose[1] for pose in mango_est]
        mango_yfinal = np.mean(mango_y)
    if len(capsicum_est) >= num_per_target:
        capsicum_x = [pose[0] for pose in capsicum_est]
        capsicum_xfinal = np.mean(capsicum_x)
        capsicum_y = [pose[1] for pose in capsicum_est]
        capsicum_yfinal = np.mean(capsicum_y) 

    for i in range(num_per_target):
        try:
            target_est['redapple_'+str(i)] = {'x':redapple_xfinal, 'y':redapple_yfinal}
        except:
            pass
        try:
            target_est['greenapple_'+str(i)] = {'x':greenapple_xfinal, 'y':greenapple_yfinal}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'x':orange_xfinal, 'y':orange_yfinal}
        except:
            pass
        try:
            target_est['mango_'+str(i)] = {'x':mango_xfinal, 'y':mango_yfinal}
        except:
            pass
        try:
            target_est['capsicum_'+str(i)] = {'x':capsicum_xfinal, 'y':capsicum_yfinal}
        except:
            pass
    ###########################################
    return target_est

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos

def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def parse_groundtruth(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            if key.startswith("redapple"):
                aruco_num = int(31)
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            if key.startswith("greenapple"):
                aruco_num = int(32)
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            if key.startswith("orange"):
                aruco_num = int(33)
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            if key.startswith("mango"):
                aruco_num = int(34)
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            if key.startswith("capsicum"):
                aruco_num = int(35)
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    
    return aruco_dict

def parse_target(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   

        fileNum = 0
        file_dict = {}

        for key in gt_dict:
            file_dict["lab_output/pred_"+str(fileNum)+".png"] = {key[:-2]:gt_dict[key]}
            fileNum = fileNum + 1
            
    return file_dict

def dict_to_vec(aruco_dict):
    aruco_vec = []
    taglist = []

    for key in aruco_dict:
        aruco_vec.append(aruco_dict[key])
        taglist.append(key)

    aruco_vec = np.hstack(aruco_vec)

    return taglist, aruco_vec

def show_map():

    ground_truth = True

    us_aruco = parse_groundtruth("true_Map.txt")
    taglist, us_vec = dict_to_vec(us_aruco)

    # Rearranging taglist
    taglist = [str(i) for i in taglist]
    for i in range(len(taglist)):
        if taglist[i] == "31":
            taglist[i] = "redapple"
        elif taglist[i] == "32":
            taglist[i] = "greenapple"
        elif taglist[i] == "33":
            taglist[i] = "orange"
        elif taglist[i] == "34":
            taglist[i] = "mango"
        elif taglist[i] == "35":
            taglist[i] = "capsicum"

    if ground_truth:
        gt_aruco = parse_groundtruth("TRUEMAP.txt")
        gt_taglist, gt_vec = dict_to_vec(gt_aruco)

        # Rearranging gt_taglist
        gt_taglist = [str(i) for i in gt_taglist]
        for i in range(len(gt_taglist)):
            if gt_taglist[i] == "31":
                gt_taglist[i] = "redapple"
            elif gt_taglist[i] == "32":
                gt_taglist[i] = "greenapple"
            elif gt_taglist[i] == "33":
                gt_taglist[i] = "orange"
            elif gt_taglist[i] == "34":
                gt_taglist[i] = "mango"
            elif gt_taglist[i] == "35":
                gt_taglist[i] = "capsicum"
    
    ax = plt.gca()
    if ground_truth:
        ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(us_vec[0,:], us_vec[1,:], marker='x', color='C1', s=100)
    for i in range(len(taglist)):
        if ground_truth:
            ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, gt_taglist[i], color='C0', size=12)
        ax.text(us_vec[0,i]+0.05, us_vec[1,i]+0.05, taglist[i], color='C1', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Truth','Pred'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # a dictionary of all the saved SLAM outputs
    target_est = {}
    with open(base_dir/'lab_output/slam.txt') as f:
        data_SLAM = f.read()
        dict_SLAM = json.loads(data_SLAM)
        taglist = dict_SLAM.get("taglist")
        coords = dict_SLAM.get("map")

        for x in range(len(taglist)):
            target_est["aruco"+str(taglist[x])+"_0"] = {'x':coords[0][x], 'y':coords[1][x]}
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    if len(image_poses) == 0:
        target_map = parse_target("lab_output/targets.txt")

    # merge the estimations of the targets so that there are only one estimate for each target type
    target_est = merge_estimations(target_map)

    # save target pose estimations
    with open(base_dir/'true_Map.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)
    
    show_map()
    