# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.06, fruit_mapping=True):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

        self.fruit_mapping = fruit_mapping

        self.marker_detected = False
        self.blind = False
    
    def detect_marker_positions(self, img, measurement_val):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error
        
        if ids is None or self.blind:
            self.marker_detected = False
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []

        for i in range(len(ids)):
            idi = ids[i,0]
            if idi < 11:
                # Some markers appear multiple times but should only be handled once.
                if idi in seen_ids:
                    continue
                else:
                    # Only update if there is 2 markers in sight during path planning
                    if len(ids) <= 1 and not self.fruit_mapping :
                        self.marker_detected = False
                        continue

                    if len(ids) == 2 and not self.fruit_mapping:
                        if (ids[0,0] == ids[1,0]):
                            self.marker_detected = False
                            continue

                    seen_ids.append(idi)
                    
                lm_tvecs = tvecs[ids==idi].T
                lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
                lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

                lm_measurement = measure.Marker(lm_bff2d, idi, measurement_val)
                measurements.append(lm_measurement)
                # print("> 2 markers")
                self.marker_detected = True
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
