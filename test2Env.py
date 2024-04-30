import numpy as np
import cv2 as cv
import os
import time
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modified_exmax_env import MobileRobot
from utils import plot_learning_curve, plot_testing_curve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Check if the method exists
if not hasattr(MobileRobot, 'show_image'):
    raise AttributeError("MobileRobot class doesn't have the 'show_image' method.")

###################################################################
#                      MANUAL CONTROL                             #
###################################################################
env = MobileRobot(2)
env.reset_pathPlanning1()
env.show_image(delay=0)  # Adjust delay time as needed
key = cv.waitKey(0)
done = False  # Initialize done as a boolean instead of a list
complete = [False, False]
action = [0, 0]
while True:
    if key == ord('7'):
        action[0] = 0
        action[1] = 0
    elif key == ord('8'):
        action[0] = 1
        action[1] = 1
    elif key == ord('9'):
        action[0] = 2
        action[1] = 2
    elif key == ord('4'):
        action[0] = 3
        action[1] = 3
    elif key == ord('6'):
        action[0] = 4
        action[1] = 4
    elif key == ord('1'):
        action[0] = 5  # Corrected from 56 to 5
        action[1] = 5
    elif key == ord('2'):
        action[0] = 6
        action[1] = 6
    elif key == ord('3'):
        action[0] = 7
        action[1] = 7
    elif key == ord('q'):
        break

    # Capture the returned values from step_pathPlanning1 method
    state, reward, done = env.step_pathPlanning1(action)

    if done:  # Check if the environment is done
        complete[0] = True
        complete[1] = True

    print('reward', reward)
    # print('done', done, 'info', info)
    print(complete)
    
    if key == ord('q'):
        break
    if complete[0] and complete[1]:
        complete[0] = False
        complete[1] = False
        env.reset_pathPlanning1()
    env.show_image(delay=0)  # Adjust delay time as needed
    key = cv.waitKey(0)
