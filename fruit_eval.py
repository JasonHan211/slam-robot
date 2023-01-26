import os

os.system("python TargetPoseEst.py")
os.system("python CV_eval.py TRUEMAP.txt lab_output/targets.txt")  