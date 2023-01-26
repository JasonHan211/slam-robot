# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time
import math
from Astar import AStarPlanner, Square
from fruit_est import FruitPosition
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot
import util.measure as measure
import util.DatasetHandler as dh # save/load functions
import pygame # python package for GUI
import shutil # python package for file operations

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector


class Operate:
    def __init__(self, args, fruit_mapping=True):
        self.folder = 'pibot_dataset/'
        self.args= args
        self.fruit_mapping = fruit_mapping
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # Initialise data parameters
        if self.args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            if fruit_mapping:
                self.pibot = Alphabot(self.args.ip, self.args.port, ir=0)
            else:
                self.pibot = Alphabot(self.args.ip, self.args.port, ir=1)

        # Covariance value
        self.measurement_val = 0.1 # 0.1 # measure
        self.wheel_val = 0.017 # 0.01 ekf

        # Initialise SLAM parameters
        self.robot = self.init_robot(self.args.calib_dir, self.args.ip)
        self.ekf = self.init_ekf(self.robot)
        self.ekf.taglist, self.ekf.markers = self.parse_groundtruth(self.args.map)
        self.ekf.markers = self.ekf.markers.T.reshape(2, 10)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06, fruit_mapping=fruit_mapping) # size of the ARUCO markers

        if self.args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if self.args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(self.args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        
        self.start = False
        self.test = False
        
        # Indexing
        self.waypoint_list = []
        self.waypoint_list_index = 1
        self.search_index = 0
        self.search_list = []
        self.waypoint = []
        
        # Speed setting
        self.speed = 1
        self.turning_vel = 15
        self.drive_vel = 25

        # Render tick
        self.renderTick = 2

        # Checking resolution (how many times rotate to make a circle)
        self.checkResolution = 8

        self.angle = np.pi/4

        self.pygame = None
        self.canvas = None

    # wheel control
    def control(self):       
        if self.args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img, self.measurement_val)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            # self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

   # Detect fruit
    def detect_fruit(self):
        self.command['inference'] = True

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # Get all seen fruit distance and angle
    def get_fruit_info(self):
        fruit_pose = FruitPosition(self.file_output[0])
        target_map = fruit_pose.get_target_info()
        return target_map

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_robot(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB_ccw = "{}baseline_ccw.txt".format(datadir)  
        baseline_ccw = np.loadtxt(fileB_ccw, delimiter=',')
        fileB_cw = "{}baseline_cw.txt".format(datadir)  
        baseline_cw = np.loadtxt(fileB_cw, delimiter=',')
        robot = Robot(baseline_ccw, baseline_cw, scale, camera_matrix, dist_coeffs)
        return robot

    # Init EKF
    def init_ekf(self, robot):
        return EKF(robot, self.wheel_val)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))

        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))
        TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))
    
    @staticmethod
    def parse_groundtruth(fname):
        with open(fname, 'r') as f:
            try:
                gt_dict = json.load(f)
            except ValueError as e:
                with open(fname, 'r') as f:
                    gt_dict = ast.literal_eval(f.readline())

            aruco_taglist = []
            aruco_marker = []
            for key in gt_dict:
                if key.startswith("aruco"):
                    aruco_taglist.append(int(key.strip('aruco')[:-2]))
                    aruco_marker.append(np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2, 1)))
        return np.array(aruco_taglist), np.array(aruco_marker)

    # Start slam
    def run_slam(self):
        n_observed_markers = len(self.ekf.taglist)
        if not self.ekf_on:
            self.notification = 'SLAM is running'
            self.ekf_on = True
        # else:
        #     self.notification = '> 2 landmarks is required for pausing'
        elif n_observed_markers < 3:
            self.notification = '> 2 landmarks is required for pausing'
        else:
            if not self.ekf_on:
                self.request_recover_robot = True
            self.ekf_on = not self.ekf_on
            if self.ekf_on:
                self.notification = 'SLAM is running'
            else:
                self.notification = 'SLAM is paused'
 
    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
             # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [self.speed, 0]
                # self.drive_robot(0.4)
                pass # TODO: replace with your M1 code to make the robot drive forward
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-self.speed, 0]
                # self.rotate_robot(-3.14)
                pass # TODO: replace with your M1 code to make the robot drive backward
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, self.speed]
                # self.rotate_robot(self.angle)
                pass # TODO: replace with your M1 code to make the robot turn left
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -self.speed]
                # self.rotate_robot(-self.angle)
                pass # TODO: replace with your M1 code to make the robot turn right
            # Decrease Speed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                self.angle = self.angle - np.pi/4
                print("- Speed: ", self.angle)
            # Increase Speed
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                self.angle = self.angle + np.pi/4
                print("+ Speed: ", self.angle)
            # W to select waypoint to go
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                operate.waypoint = operate.click_for_waypoint()
            # Keyup to stop moving
            elif event.type == pygame.KEYUP and (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN):
                self.command['motion'] = [0,0]
            # start path planning
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                self.start = True
             # start path planning
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                if self.test:
                    self.test = False
                else:
                    self.test = True
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # self.command['motion'] = [0, 0]
                self.check_position_with_vision(self.pygame,self.canvas)
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                
                os.system("python fruit_eval.py")
                os.system("python create_map.py")
                
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.run_slam()
            # run fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                # self.detect_fruit()
                self.scan_fruit()
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    # scan for fruit
    def scan_fruit(self):

        self.take_pic()
        self.detect_fruit()
        self.detect_target()
        map = self.get_fruit_info()

        # visualise
        self.draw(self.canvas)
        self.pygame.display.update()

        return map

    # Render GUI
    def render(self):

        self.update_keyboard()
        self.take_pic()
        drive_meas = self.control()
        self.update_slam(drive_meas) 
        self.record_data()
        self.save_image()
        
        # Print robot pose
        if self.fruit_mapping:
            print(self.ekf.robot.state.flatten('F').round(3))

        # visualise
        self.draw(self.canvas)
        self.pygame.display.update()

    # Rotate robot
    def rotate_robot(self, turnAngle):
        
        # time.sleep(1)
        turn_time = turn_angle_to_time(self.turning_vel, turnAngle, self.ekf.robot)
        print("Start rot @ time:", round(turn_time,3), "angle:", round(np.degrees(turnAngle),3))
        if turnAngle > 0:
            lv,rv = self.pibot.set_velocity([0, 1], turning_tick=self.turning_vel, time=turn_time)
        else:
            lv,rv = self.pibot.set_velocity([0, -1], turning_tick=self.turning_vel, time=turn_time)
        drive_meas = measure.Drive(lv, rv, turn_time)
        self.update_slam(drive_meas)

    # Drive robot
    def drive_robot(self, distance):
        
        # time.sleep(1)
        drive_time = distance_to_time(self.drive_vel, distance)
        print("Start drive @ time:", round(drive_time,3), "distance:", round(distance,3))
        lv,rv = self.pibot.set_velocity([1, 0], tick=self.drive_vel, time=drive_time)
        drive_meas = measure.Drive(lv, rv, drive_time)
        self.update_slam(drive_meas)

    # Check position with vision
    def check_position_with_vision(self):
        
        self.aruco_det.blind = False
        
        resolution = self.checkResolution
        print("\nChecking position...")
        time.sleep(1)

        seen = False
        count = 0
        while not seen:
            if count % 8 == 0 and not count == 0:
                print("Shift to see")
                self.drive_robot(0.1)
            
            angle = (2*np.pi)/resolution
            self.rotate_robot(angle)
            time.sleep(0.5)
            self.render()
            seen = self.aruco_det.marker_detected
            if seen:
                # let it converge
                tick = 110
                while tick > 0:
                    print(tick)
                    self.render()
                    tick = tick - 1
                self.aruco_det.blind = True
                print("Checking position done")
                time.sleep(2)
                return
            count = count + 1

    # Check position with vision
    def check_reach_fruit(self):

        print("\nChecking fruit ...")
        
        time.sleep(1)

        reach = False
        
        for i in range(8):
            print("Scan",i)

            # Scan for fruit
            self.render()
            dict = self.scan_fruit()
            time.sleep(1)
            print(dict)
            self.render()

            # Get target fruit from key
            for key in dict:
                if self.search_list[self.search_index] == key:
                    info = dict[key]
                    distance = info['distance']
                    angle = info['angle']

                    print(key,distance,angle)

                    if distance <= 0.4:
                        reach = True
                        print("Done checking fruit")
                        return reach
                    else:
                        if abs(angle) > 0.35:
                            self.rotate_robot(angle)

                        if distance > 0.4 and distance < 1.2:
                            self.drive_robot(distance - 0.35)
                            reach = True

                        self.render()
                        return reach
                
            if not reach:
                print("Turning right")
                self.rotate_robot(-0.785398)
                time.sleep(1)
                self.render()

        print("No fruit in sight")
        return reach

    # Check reach fruit with position
    def check_reach_fruit_with_position(self):
        
        self.check_position_with_vision()
        dist, _ = get_distance_angle(operate.waypoint_list[-1], operate.ekf.robot.state)

        if dist < 0.4:
            return True
        else:
            return False

    # Get current waypoint
    def get_waypoint(self):
        if self.waypoint_list_index >= len(self.waypoint_list):
            return []
        return self.waypoint_list[self.waypoint_list_index]

    # Get map plot
    def get_map(self):
        fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(self.args.map)
        search_list = read_search_list()

        # Initialize
        gx = []
        gy = []
        ox = []
        oy = []

        foy = []
        fox = []

        sx = 0
        sy = 0
        dx = 0
        dy = 0

        if self.search_index >= len(search_list):
            return ox,oy,sx,sy,dx,dy,gx,gy

        print("\n\nSearching: ",search_list[self.search_index])

        # goal
        for j in range(len(fruits_list)):
            if search_list[self.search_index] == fruits_list[j]:
                gx.append(int(fruits_true_pos[j, 0] * 10) + 16)
                gy.append(int(fruits_true_pos[j, 1] * 10) + 16)

        # fruit obstacle list
        fruit_obstacle = fruits_list.copy()
        fruit_obstacle.remove(search_list[self.search_index])

        # print("Unwanted fruit",fruit_obstacle)

        for i in range(len(fruit_obstacle)):
            for j in range(len(fruits_list)):
                if fruit_obstacle[i] == fruits_list[j]:
                    fox.append(fruits_true_pos[j, 0])
                    foy.append(fruits_true_pos[j, 1])

        # obstacle generation
        for i in range(len(fruit_obstacle)):
            a, b = Square(int(fox[i] * 10) + 16, int(foy[i] * 10) + 16, 1)
            ox = ox + a
            oy = oy + b    

        for i in range(aruco_true_pos.shape[0]):
            a, b = Square(int(aruco_true_pos[i, 0] * 10) + 16, int(aruco_true_pos[i, 1] * 10) + 16, 1)
            ox = ox + a
            oy = oy + b

        # start pos
        sx = (self.ekf.robot.state[0] * 10) + 16
        sy = (self.ekf.robot.state[1] * 10) + 16

        dx = sx + (1 * np.cos(self.ekf.robot.state[2])) 
        dy = sy + (1 * np.sin(self.ekf.robot.state[2])) 

        return ox,oy,sx,sy,dx,dy,gx,gy

    # Get waypoint from map
    def click_for_waypoint(self):

        # Get map
        ox,oy,sx,sy,dx,dy,gx,gy = self.get_map()

        fig = plt.figure()
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(dx, dy, ".r")
        plt.plot(gx, gy, "xb")
        space = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
        plt.xticks(space)
        plt.yticks(space)
        plt.grid(True)
        plt.axis("equal")

        # Variables, p will contains clicked points, idx contains current point that is being selected
        global point 
        
        point = np.ones((1,2))
        point[0,0] = operate.ekf.robot.state[0,0]
        point[0,1] = operate.ekf.robot.state[1,0]
        
        # pick points
        def onclick(event):
            
            if event.button == 1:
                # left mouse click, add point and increment by 1
                point[0,0] = (event.xdata - 16)/10
                point[0,1] = (event.ydata - 16)/10

            print(str(point))
        
        print("\nClick on the waypoint for the robot to move\n")
        plt.get_current_fig_manager().set_window_title('Close window once point is selected')    
        ka = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return point[0]

    # Get next waypoint list
    def search_next(self):
        self.waypoint_list = []
        
        ox,oy,sx,sy,dx,dy,gx,gy = self.get_map()
        
        grid_size = 1
        robot_radius = 1

        if len(gx) == 0:
            return

        astar = AStarPlanner(ox, oy, grid_size, robot_radius)
        rx, ry = astar.planning(sx[0], sy[0], gx[0], gy[0])

        # Plot path
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(dx, dy, "*r")
        plt.plot(gx, gy, "xb")
        space = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
        plt.xticks(space)
        plt.yticks(space)
        plt.grid(True)
        plt.axis("scaled")
        plt.xlim([0,32])
        plt.ylim([0,32])
        u = np.diff(rx)
        v = np.diff(ry)
        pos_x = rx[:-1] + u/2
        pos_y = ry[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        plt.plot(rx, ry, "--r")
        plt.quiver(pos_x, pos_y, -u/norm, -v/norm, angles="xy", zorder=5, width=0.004, headwidth=4, headlength=4, pivot="mid", color='r')
        plt.pause(0.001)
        # plt.show()
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        rx = (np.array(rx) - 16) / 10
        ry = (np.array(ry) - 16) / 10
        waypoint = []
        for i in range(len(rx)):
            waypoint.append(np.array([rx[i], ry[i]]))
        self.waypoint_list = np.flip(np.array(waypoint), axis=0)
        print("Waypoint List: ")
        print(self.waypoint_list)
        print(len(waypoint))

        if len(waypoint) == 1:
            return False
        else:
            return True
        

# Read true map
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

# Read search list
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

# Print fruit position
def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

# Calculate turn time given angle
def turn_angle_to_time(turning_vel, turnAngle, robot):
    # Import files
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB_ccw = "calibration/param/baseline_ccw.txt"
    fileB_cw = "calibration/param/baseline_cw.txt"

    # ccw
    if turnAngle > 0:
        # baseline = np.loadtxt(fileB_ccw, delimiter=',')
        baseline = -0.0089*(abs(turnAngle)**3) + 0.0516*(abs(turnAngle)**2) - 0.09*(abs(turnAngle)) + 0.1605
        robot.wheels_width_ccw = baseline
    # cw
    else:
        # baseline = np.loadtxt(fileB_cw, delimiter=',')
        baseline = -0.0277*(abs(turnAngle)**3) + 0.1673*(abs(turnAngle)**2) - 0.3095*(abs(turnAngle)) + 0.2951
        robot.wheels_width_cw = baseline
    
    turn_time = (baseline * abs(turnAngle)) / (scale * turning_vel * 2)

    return turn_time

# Calculate drive time given distance
def distance_to_time(wheel_vel, distance):
    # Import files
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')

    drive_time = distance/(scale * wheel_vel)

    return drive_time

# Calculate distanca and angle given waypoint and robot state
def get_distance_angle(waypoint, robot_pose):

    dy = (waypoint[1] - robot_pose[1])
    dx = (waypoint[0] - robot_pose[0])

    distance = math.hypot(dx, dy)

    theta = math.atan2(dy, dx)

    robot_angle = robot_pose[2][0]

    # print("\nPre robot angle",np.degrees(robot_angle))

    if robot_angle > 2*np.pi or robot_angle < -2*np.pi:
        robot_angle = robot_angle % (2*np.pi)

    if robot_angle > np.pi and robot_angle > 0:
        robot_angle = -2*np.pi + robot_angle
    elif robot_angle < -np.pi and robot_angle < 0:
        robot_angle = 2*np.pi + robot_angle

    # print("Theta",np.degrees(theta),"Robot angle",np.degrees(robot_angle),"\n")

    turnAngle = theta - robot_angle

    if math.isclose(turnAngle,2*np.pi,abs_tol=0.05):
        turnAngle = 0
    elif math.isclose(turnAngle,-2*np.pi,abs_tol=0.05):
        turnAngle = 0
    elif math.isclose(turnAngle, np.pi, abs_tol=0.05):
        turnAngle = np.pi
    elif turnAngle > np.pi and turnAngle > 0:
        turnAngle = -2*np.pi + turnAngle
    elif turnAngle < -np.pi and turnAngle < 0:
        turnAngle = 2*np.pi + turnAngle

    return  distance, turnAngle


# main loop
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='true_Map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.0.13')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()

    pygame.font.init()

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                    pygame.image.load('pics/8bit/pibot2.png'),
                    pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                    pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    # Loading page
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    # Choose level 0, 1 or 2
    # level 0 = fruit mapping
    # level 1 = click waypoints
    # level 2 = path planning

    level = 0
    use_vision = False

    fruit_mapping = False
    if level == 0:
        fruit_mapping = True

    # Start slam
    operate = Operate(args,fruit_mapping=fruit_mapping)

    operate.pygame = pygame
    operate.canvas = canvas

    operate.run_slam()

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    operate.search_list = search_list
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    while True:

        operate.render()

        if not level == 0:

            # If ekf (slam) is on
            if operate.ekf_on:

                if operate.start:

                    # Get waypoints
                    # Level 1
                    if level == 1:

                        # Auto need for selecting waypoint or
                        # Press W on keyboard to select waypoint
                        operate.waypoint = operate.click_for_waypoint()

                    # Level 2
                    elif level == 2:

                        # Get list on the first time
                        if len(operate.waypoint_list) == 0:
                            exist = operate.search_next()

                        # Get waypoint from list
                        operate.waypoint = operate.get_waypoint()

                    # If there is waypoint
                    if len(operate.waypoint) != 0:

                        # Check if final destination is within
                        dist, _ = get_distance_angle(operate.waypoint_list[-1], operate.ekf.robot.state)

                        if dist > 0.3:
                            # Distance and angle to waypoint
                            distance, angle = get_distance_angle(operate.waypoint, operate.ekf.robot.state)
                            print("\n=======================================================")
                            print("Waypoint:",operate.waypoint.round(3),"Robot:",operate.ekf.robot.state.flatten('F').round(3))
                            print("Robot Angle",round(np.degrees(operate.ekf.robot.state[2][0]),3)%360)
                            print("Distance:",round(distance,3),"Angle:",round(np.degrees(angle),3))
                            print("=======================================================\n")

                            # Turn robot

                            operate.render()
                            
                            operate.rotate_robot(angle)

                            operate.render()

                            # Get robot's pose and waypoint
                            distance, angle = get_distance_angle(operate.waypoint, operate.ekf.robot.state)
                            print("After rot = dist:",round(distance,3),"angle:",round(np.degrees(angle),3))


                        # Check distance to fruit if within 0.3 m
                        if level == 2:
                            dist, _ = get_distance_angle(operate.waypoint_list[-1], operate.ekf.robot.state)
                            
                            if dist < 0.3:

                                time.sleep(1)

                                if not use_vision:
                                    reach = operate.check_reach_fruit_with_position()
                                else:
                                    reach = operate.check_reach_fruit()

                                if reach:
                                    print("\n\tIm at the fruit!!!\n")
                                    time.sleep(3)
                                    operate.search_index = operate.search_index + 1
                                else:
                                    print("\n\tFruit not in radius\n")

                                if use_vision:
                                    operate.check_position_with_vision()

                                exist = False
                                while not exist:
                                    exist = operate.search_next()
                                    if not exist:
                                        operate.drive_robot(0.1)
                                
                                operate.waypoint_list_index = 1

                                continue

                        # Drive robot
                        operate.render()

                        cap_distance = False
                        # Cap driving distance
                        if distance > 0.3 and level == 2:
                            cap_distance = True
                            distance = 0.4
                        
                        operate.drive_robot(distance)
                        time.sleep(0.5)

                        operate.render()

                        # Get robot's pose and waypoint
                        distance, angle = get_distance_angle(operate.waypoint, operate.ekf.robot.state)
                        print("After drive = dist:",round(distance,3),"angle:",round(np.degrees(angle),3))
                        print("\nAfter traverse Robot pose:",operate.ekf.robot.state.flatten('F').round(3),"\n")
                        
                        if level == 1:
                            operate.waypoint = []
                        if level == 2:
                            if cap_distance:
                                operate.check_position_with_vision()
                            else:
                                operate.waypoint_list_index = operate.waypoint_list_index + 1
                    else:
                        print("No waypoint")
                        operate.start = False
                        operate.run_slam()

                # Testing playground
                if operate.test:

                    # Set robot pose
                    robot_pose = np.array([[0.5807404431506688], [0.6541118668899104], [35.85228088310865]])
                    operate.ekf.robot.state = robot_pose

                    operate.search_index = 2
                    exist = False
                    while not exist:
                        exist = operate.search_next()
                        if not exist:
                            operate.drive_robot(0.1)


                    ''' Notes
                    DW Battery 1 and 2
                    


                    '''

                    # End playground
                    operate.test = False
        
