# for computing the wheel calibration parameters
import numpy as np
import os
import sys
sys.path.insert(0, "../util")
from pibot import Alphabot

ccw = np.zeros((2,4))
cw = np.zeros((2,4))

def calibrateWheelRadius():
    # Compute the robot scale parameter using a range of wheel velocities.
    # For each wheel velocity, the robot scale parameter can be computed
    # by comparing the time and distance driven to the input wheel velocities.

    ##########################################
    # Feel free to change the range / step
    ##########################################
    # wheel_velocities_range = range(15, 30, 5)  # or use np.linspace
    wheel_velocities_range = [25]  # or use np.linspace
    delta_times = []

    for wheel_vel in wheel_velocities_range:
        print("Driving at {} ticks/s.".format(wheel_vel))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Drive the robot at the given speed for the given time
            ppi.set_velocity([1, 0], tick=wheel_vel, time=delta_time)

            uInput = input("Did the robot travel 1m?[y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot drove 1m in {:.2f} seconds at wheel speed {}.\n".format(delta_time,
                                                                                                        wheel_vel))
                break

    # Once finished driving, compute the scale parameter by averaging
    num = len(wheel_velocities_range)
    scale = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        pass # TODO: replace with your code to compute the scale parameter using wheel_vel and delta_time
        scale = scale + (1/(wheel_vel*delta_time))
    scale = scale/num
    print("The scale parameter is estimated as {:.6f} m/ticks.".format(scale))

    return scale


def calibrateBaseline(scale, angle, turn_cw=True):
    # Compute the robot baseline parameter using a range of wheel velocities.
    # For each wheel velocity, the robot baseline parameter can be computed by
    # comparing the time elapsed and rotation completed to the input wheel
    # velocities to find out the distance between the wheels (wheels_width).

    ##########################################
    # Feel free to change the range / step
    ##########################################
    # wheel_velocities_range = range(15, 30, 5)  # or use np.linspace
    wheel_velocities_range = [15]  # or use np.linspace
    delta_times = []

    for wheel_vel in wheel_velocities_range:
        print("Spinning at {} ticks/s. {}\n".format(wheel_vel,angle))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to spin in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Spin the robot at the given speed for the given time
            if turn_cw:
                ppi.set_velocity([0, -1], turning_tick=wheel_vel, time = delta_time)
            else:
                ppi.set_velocity([0, 1], turning_tick=wheel_vel, time = delta_time)

            uInput = input("Did the robot spin deg?[y/N]")
            if uInput == 'y':
                
                delta_times.append(delta_time)
                print("Recording that the robot spun ",angle,"deg in {:.2f} seconds at wheel speed {}.\n".format(delta_time,
                                                                                                           wheel_vel))
                break
    
    x = 360/angle

    # Once finished driving, compute the baseline parameter by averaging
    num = len(wheel_velocities_range)
    baseline = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_vel, and delta_time
        baseline = baseline + ((scale * wheel_vel * delta_time * x)/ np.pi)
    baseline = baseline/num

    index = int((angle/45)) - 1
    print("index",index)
    if turn_cw:
        cw[0,index] = delta_time
        cw[1,index] = baseline
    else:
        ccw[0,index] = delta_time
        ccw[1,index] = baseline

    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.0.13')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port,ir=0)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    print('Calibrating PiBot scale...\n')
    scale = calibrateWheelRadius()
    fileNameS = "{}scale.txt".format(dataDir)
    np.savetxt(fileNameS, np.array([scale]), delimiter=',')

    # print('Calibrating PiBot ccw baseline...45\n')
    # baseline_45_ccw = calibrateBaseline(scale,45,turn_cw=False)
    # fileNameB = "{}baseline_ccw.txt".format(dataDir)
    # np.savetxt(fileNameB, np.array([baseline_45_ccw]), delimiter=',')

    # print('Calibrating PiBot ccw baseline...90\n')
    # baseline_90_ccw = calibrateBaseline(scale,90,turn_cw=False)
    # fileNameB = "{}baseline_ccw.txt".format(dataDir)
    # np.savetxt(fileNameB, np.array([baseline_90_ccw]), delimiter=',')

    # print('Calibrating PiBot ccw baseline...135\n')
    # baseline_135_ccw = calibrateBaseline(scale,135,turn_cw=False)
    # fileNameB = "{}baseline_ccw.txt".format(dataDir)
    # np.savetxt(fileNameB, np.array([baseline_135_ccw]), delimiter=',')

    # print('Calibrating PiBot ccw baseline...180\n')
    # baseline_180_ccw = calibrateBaseline(scale,180,turn_cw=False)
    # fileNameB = "{}baseline_ccw.txt".format(dataDir)
    # np.savetxt(fileNameB, np.array([baseline_180_ccw]), delimiter=',')

    print('Calibrating PiBot cw baseline...45\n')
    baseline_45_cw = calibrateBaseline(scale,45,turn_cw=True)
    fileNameB = "{}baseline_cw.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline_45_cw]), delimiter=',')

    print('Calibrating PiBot cw baseline...90\n')
    baseline_90_cw = calibrateBaseline(scale,90,turn_cw=True)
    fileNameB = "{}baseline_cw.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline_90_cw]), delimiter=',')

    print('Calibrating PiBot cw baseline...135\n')
    baseline_135_cw = calibrateBaseline(scale,135,turn_cw=True)
    fileNameB = "{}baseline_cw.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline_135_cw]), delimiter=',')

    print('Calibrating PiBot cw baseline...180\n')
    baseline_180_cw = calibrateBaseline(scale,180,turn_cw=True)
    fileNameB = "{}baseline_cw.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline_180_cw]), delimiter=',')

    print('\nFinished calibration\n')

    print("ccw: ",ccw)
    print("cw: ",cw)

