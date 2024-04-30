import numpy as np
import math
import cv2 as cv
import random
import gymnasium as gym
from gymnasium import spaces

class MobileRobotEnv(gym.Env):
    def __init__(self, n_robots):
        super(MobileRobotEnv, self).__init__()
        # Parameters for path planning
        self.mapW = 40
        self.mapH = 24
        self.span_x = 37
        self.span_y = 21
        self.gridW = self.mapW + self.span_x * 2
        self.gridH = self.mapH + self.span_y * 2
        self.robot_rad = 1
        self.group_rad = self.robot_rad * 3
        self.group_gap = self.group_rad - self.robot_rad
        self.n_robots = n_robots
        self.action_space = spaces.Discrete(8)  # 8 possible actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.n_robots, 3, 3), dtype=np.uint8)
        self.robots_pos = [[0, 0]] * n_robots  # Placeholder for robot positions
        self.map_image = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)  # Placeholder for map image

    def reset(self):
        # Reset environment to initial state
        self.robots_pos = self._generate_initial_positions()
        self.map_image = self._generate_map_image()
        return self._get_observation()

    def step(self, actions):
        # Perform actions and update environment
        for k in range(self.n_robots):
            self._move_robot(k, actions[k])
        obs = self._get_observation()
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        # Render the environment
        img_resized = cv.resize(self.map_image, (self.gridW * 10, self.gridH * 10), interpolation=cv.INTER_NEAREST)
        cv.imshow('Mobile Robot Environment', img_resized)
        cv.waitKey(1)
        return img_resized

    def _generate_initial_positions(self):
        # Generate initial positions for robots
        return [[random.randint(0, self.gridW - 1), random.randint(0, self.gridH - 1)] for _ in range(self.n_robots)]

    def _generate_map_image(self):
        # Generate a visual representation of the map
        map_image = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        # Placeholder code for visualization (you can modify this based on your actual implementation)
        for pos in self.robots_pos:
            map_image[pos[1], pos[0]] = [255, 0, 0]  # Red for robots
        return map_image

    def _move_robot(self, idx, action):
        # Move a robot based on the chosen action
        dx, dy = 0, 0
        if action == 0:  # move up-left
            dx, dy = -1, -1
        elif action == 1:  # move up
            dy = -1
        elif action == 2:  # move up-right
            dx, dy = 1, -1
        elif action == 3:  # move left
            dx = -1
        elif action == 4:  # move right
            dx = 1
        elif action == 5:  # move down-left
            dx, dy = -1, 1
        elif action == 6:  # move down
            dy = 1
        elif action == 7:  # move down-right
            dx, dy = 1, 1

        new_x = max(0, min(self.gridW - 1, self.robots_pos[idx][0] + dx))
        new_y = max(0, min(self.gridH - 1, self.robots_pos[idx][1] + dy))
        self.robots_pos[idx] = [new_x, new_y]

    def _get_observation(self):
        # Generate observation for the current state
        observation = np.zeros((self.n_robots, 3, 3, 3), dtype=np.uint8)
        for k in range(self.n_robots):
            x, y = self.robots_pos[k]
            # Ensure the slice covers a 3x3 region around the robot
            x_min = max(0, x - 1)
            x_max = min(self.gridW, x + 2)
            y_min = max(0, y - 1)
            y_max = min(self.gridH, y + 2)
            observation[k, :y_max - y_min, :x_max - x_min] = self.map_image[y_min:y_max, x_min:x_max]
        return observation



# Example usage
env = MobileRobotEnv(n_robots=2)
obs = env.reset()

for _ in range(100):
    actions = [env.action_space.sample() for _ in range(env.n_robots)]
    obs, reward, done, info = env.step(actions)
    env.render()
