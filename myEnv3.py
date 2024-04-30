import numpy as np
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces
import math
import random

class CustomEnv(gym.Env):
    """Custom Environment that follows the gym interface, specifically for a mobile robot simulation."""

    metadata = {'render_modes': ['human'], 'render_fps': 120}

    def __init__(self, n_robots=2):  # Set to 2 robots as per scenario
        super(CustomEnv, self).__init__()
        self.n_robots = n_robots
        self.mobile_robot = MobileRobot(n_robots)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 1000, 3), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([8] * n_robots)
        self.state = None
        self.rewards_collected = [False] * n_robots  # Track rewards for each robot

    def step(self, actions):
        self.state, rewards, terminals, infos = self.mobile_robot.step_pathPlanning1(actions)

        # Update reward collection status and aggregate total reward
        total_reward = 0
        for i, reward in enumerate(rewards):
            if reward > 0:
                self.rewards_collected[i] = True
            total_reward += reward

        terminated = all(self.rewards_collected)  # Game ends when all robots collect rewards
        truncated = False
        info = {'individual_rewards': rewards, 'other_info': infos}
        observation = self.render()
        
        return observation, total_reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mobile_robot.reset_pathPlanning1()
        self.state = self.mobile_robot.get_stackSet()
        self.rewards_collected = [False] * self.n_robots  # Reset reward status
        observation = self.render()
        return observation, {}

    def render(self, mode='human'):
        img = self.mobile_robot.render_pathPlanning1()
        img_resized = cv.resize(img, (1000, 600))
        return img_resized

    def close(self):
        cv.destroyAllWindows()

class MobileRobot(object):
    def __init__(self, n_robots):
        self.initialize_params(n_robots)

    def initialize_params(self, n_robots):
        self.n_robots = n_robots
        # Define other parameters, including reward locations

    def reset_pathPlanning1(self):
        # Reset the state for a new episode
        return self.get_stackSet()

class MobileRobot(object):
    def __init__(self, n_robots):
        # Parameters for path planning
        self.mapW = 40
        self.mapH = 24
        self.span_x = 37
        self.span_y = 21
        self.gridW = self.mapW + self.span_x*2
        self.gridH = self.mapH + self.span_y*2
        self.robot_rad = 1
        self.group_rad = self.robot_rad*3
        self.group_gap = self.group_rad - self.robot_rad
        self.n_robots = n_robots
        self.n_stacks = 4
        self.n_actions_pathPlanning = 8
        self.n_actions_motionControl = 2
        self.n_obstacles = 3
        self.obs_size_min = 2
        self.obs_size_max = 4
        self.img_ratio = 50
        self.map_ratio = 4
        self.input_dims_pathPlanning = (self.n_stacks, (self.span_y*2 + 1)*self.map_ratio, (self.span_x*2 + 1)*self.map_ratio)
        self.input_dims_motionControl = 2
        self.robot_width = 120
        self.axis_length = self.robot_width
        self.max_actions_motionControl = [100, 1.0]
    
        # Colors
        self.floor_color = [255, 255, 255]
        self.obstacle_color = [100, 100, 100]
        self.dropOff_color = [0, 255, 0]
        self.loading_color = [0, 255, 255]
        self.path_center_color = [0, 255, 255]
        self.robots_color = [[255, 128, 0], [128, 0, 255], [0, 128, 255], [0, 0, 255], [0, 64, 128]]
        self.paths_color = [[255, 182, 108], [190, 125, 255], [90, 170, 255], [104, 104, 255], [64, 128, 128]]
        self.err_color = [[246, 130, 50], [80, 134, 240], [34, 125, 55]]
        self.wheel_color = [0, 0, 0]
        self.font_color = [0, 0, 0]


    def reset_pathPlanning1(self):
        self.create_obstacles()
        self.get_gridGlobal2()
        self.create_loading_point()
        self.create_dropOff_point()
        self.get_gridGlobal1()
        self.create_goals()
        self.create_starts()
        self.get_gridSet()
        self.robots_pos = self.starts.copy()
        self.get_paths()
        self.get_stateSet()
        stack_set = self.get_stackSet()
        return stack_set

    def create_obstacles(self):
        while True:
            self.obstacles = []
            for _ in range(100):
                w = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                h = random.randrange(self.obs_size_min, self.obs_size_max + 1)
                x = random.randrange(self.span_x, self.gridW - self.span_x - w + 1)
                y = random.randrange(self.span_y, self.gridH - self.span_y - h + 1)
                if not self.overlap_obstacle((x, y, w, h)):
                    self.obstacles.append([x, y, w, h])
                if len(self.obstacles) == 3:
                    break
            if len(self.obstacles) == 3:
                break

    def overlap_obstacle(self, obs):
        x, y, w, h = obs
        for x_, y_, w_, h_  in self.obstacles:
            if x in range(x_ - w - self.group_rad*2, x_ + w_ + self.group_rad*2 + 1) and \
                y in range(y_ - h - self.group_rad*2, y_ + h_  + self.group_rad*2 + 1):
                return True
        return False

    def get_gridGlobal2(self):
        self.grid_global2 = np.zeros((self.gridH, self.gridW))
        for i in range(self.gridW):
            for j in range(self.gridH):
                if i not in range(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad) or \
                    j not in range(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad):
                    self.grid_global2[j, i] = -1
                else:  
                    for x, y, w, h in self.obstacles:
                        if i in range(x - self.group_rad, x + w + self.group_rad) and \
                            j in range(y - self.group_rad, y + h + self.group_rad):
                            self.grid_global2[j, i] = -1

    def create_loading_point(self):
        while True: 
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0:
                self.grid_global2[y,x] = 2
                self.loading_point = [x,y]
                break

    def create_dropOff_point(self):
        while True:
            x = random.randrange(self.span_x + self.group_rad, self.gridW - self.span_x - self.group_rad)
            y = random.randrange(self.span_y + self.group_rad, self.gridH - self.span_y - self.group_rad)
            if self.grid_global2[y,x] == 0 and self.cal_distance([x,y], self.loading_point) > (self.mapH + self.mapW)/4:
                self.grid_global2[y,x] = 1
                self.dropOff_point = [x,y]
                break

    def get_gridGlobal1(self):
        self.grid_global1 = np.zeros((self.gridH, self.gridW))
        for i in range(self.gridW):
            for j in range(self.gridH):
                if i not in range(self.span_x + self.robot_rad, self.gridW - self.span_x - self.robot_rad) or \
                    j not in range(self.span_y + self.robot_rad, self.gridH - self.span_y - self.robot_rad):
                    self.grid_global1[j, i] = -1
                else:             
                    for x, y, w, h in self.obstacles:
                        if i in range(x - self.robot_rad, x + w + self.robot_rad) and \
                            j in range(y - self.robot_rad, y + h + self.robot_rad):
                            self.grid_global1[j, i] = -1
        
    def create_goals(self):
        # set goal values
        self.goals = []
        self.img_load_pts = []
        self.img_drop_pts = []
        theta = 2*math.pi/self.n_robots
        for i in range(self.n_robots):
            img_x = int((self.loading_point[0] - self.span_x + self.group_gap*math.cos(theta*i))*self.img_ratio + self.img_ratio//2)
            img_y = int((self.loading_point[1] - self.span_y + self.group_gap*math.sin(theta*i))*self.img_ratio + self.img_ratio//2)
            img_x_ = int((self.dropOff_point[0] - self.span_x + self.group_gap*math.cos(theta*i))*self.img_ratio + self.img_ratio//2)
            img_y_ = int((self.dropOff_point[1] - self.span_y + self.group_gap*math.sin(theta*i))*self.img_ratio + self.img_ratio//2)
            self.img_load_pts.append([img_x, img_y])
            self.img_drop_pts.append([img_x_, img_y_])
            x = int(img_x//self.img_ratio + self.span_x)
            y = int(img_y//self.img_ratio + self.span_y)
            x_ = int(img_x_//self.img_ratio + self.span_x)
            y_ = int(img_y_//self.img_ratio + self.span_y)
            self.goals.append([x,y])
            self.grid_global1[y,x] = i+1 
            self.grid_global1[y_,x_] = i+21 

    def create_starts(self):
        self.starts = []
        # find start position for robot1
        while True:
            x = random.randrange(self.span_x + self.robot_rad + 25, self.gridW - self.span_x - self.robot_rad)
            y = random.randrange(self.span_y + self.robot_rad, self.gridH - self.span_y - self.robot_rad)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 11
                break
        # find start position for robot2
        while True:
            x = random.randrange(self.span_x + self.robot_rad, self.span_x + self.robot_rad + 25)
            y = random.randrange(self.span_y + self.robot_rad + 11, self.gridH - self.span_y - self.robot_rad)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 12
                break
        # find start position for robot3
        while True:
            x = random.randrange(self.span_x + self.robot_rad, self.span_x + self.robot_rad + 25)
            y = random.randrange(self.span_y + self.robot_rad, self.span_y + self.robot_rad + 11)
            if self.grid_global1[y, x] == 0 and self.available_position([x, y]):
                self.starts.append([x, y])
                self.grid_global1[y,x] = 13
                break
    
    def available_position(self, point):
        for start_pos in self.starts:
            if self.cal_distance(point, start_pos) < self.robot_rad*5:
                return False
        return True

    def get_gridSet(self):
        self.grid_set = []
        for k in range(self.n_robots):
            grid = np.zeros((self.gridH, self.gridW))
            for i in range(self.gridW):
                for j in range(self.gridH):
                    if self.grid_global1[j,i] == -1:
                        grid[j,i] = -1
                    elif self.grid_global1[j,i] == k+1:
                        grid[j,i] = 1
                    elif self.grid_global1[j,i] == k+11:
                        grid[j,i] = 2
            self.grid_set.append(grid)

    def get_paths(self):
        self.paths = []
        for k in range(self.n_robots):
            path = [self.robots_pos[k]]
            self.paths.append(path)

    def get_stateSet(self):
        self.state_set = []
        for k in range(self.n_robots):
            state = self.grid_set[k][self.robots_pos[k][1] - self.span_y: self.robots_pos[k][1] + self.span_y + 1,
                                    self.robots_pos[k][0] - self.span_x: self.robots_pos[k][0] + self.span_x + 1]
            state_resized = cv.resize(state, ((self.span_x*2 + 1)*self.map_ratio, (self.span_y*2 + 1)*self.map_ratio), 
                                    interpolation=cv.INTER_AREA)
            self.state_set.append(state_resized)

    def get_stackSet(self):
        self.stack_set = []
        stack_set = []
        for k in range(self.n_robots):
            stack = [self.state_set[k].copy()]*self.n_stacks
            self.stack_set.append(stack)
            stack_set.append(np.array(stack))
        return stack_set

    def step_pathPlanning1(self, actions):
        new_positions = []
        rewards = []
        terminals = []
        infos = []
        for k in range(self.n_robots):
            new_position = self.robots_pos[k].copy()
            if actions[k] == 0: # move up-left
                new_position[0] -= 1
                new_position[1] -= 1
            elif actions[k] == 1: # move up
                new_position[1] -= 1
            elif actions[k] == 2: # move up-right
                new_position[0] += 1
                new_position[1] -= 1
            elif actions[k] == 3: # move left
                new_position[0] -= 1
            elif actions[k] == 4: # move right
                new_position[0] += 1
            elif actions[k] == 5: # move down-left
                new_position[0] -= 1
                new_position[1] += 1
            elif actions[k] == 6: # move down
                new_position[1] += 1
            elif actions[k] == 7: # move down-right
                new_position[0] += 1
                new_position[1] += 1

            dist = self.cal_distance(self.robots_pos[k], self.goals[k]) - self.cal_distance(new_position, self.goals[k])
            line = self.cal_distance(new_position, self.robots_pos[k])
            reward = dist -line
            terminal = False
            info = ''

            if self.reachGoal(new_position, k):
                reward = 0
                terminal = True
                info = 'reach goal'
            elif self.HitObstacle(new_position, self.grid_set[k]):
                reward = -3
                new_position = self.robots_pos[k].copy()
                info = 'hit obstacle'

            new_positions.append(new_position)
            rewards.append(reward)
            terminals.append(terminal)
            infos.append(info)
        
        self.update_gridSet(new_positions)
        self.add_paths()
        self.get_stateSet()
        stack_set = self.update_stackSet()

        return stack_set, rewards, terminals, infos

    def reachGoal(self, position, idx):
        if position == self.goals[idx]:
            return True
        else:
            return False

    def HitObstacle(self, position, grid):
        if grid[position[1], position[0]] == -1:
            return True
        else:
            return False

    def update_gridSet(self, new_positions):
        for k in range(self.n_robots):
            # update grid global1
            self.grid_global1[self.robots_pos[k][1], self.robots_pos[k][0]] = 0
            self.grid_global1[new_positions[k][1], new_positions[k][0]] = k+11
            # update grid set
            self.grid_set[k][self.robots_pos[k][1], self.robots_pos[k][0]] = 0
            self.grid_set[k][new_positions[k][1], new_positions[k][0]] = 2
            # update robot's positions
            self.robots_pos[k] = new_positions[k]

    def add_paths(self):
        for k in range(self.n_robots):
            if self.robots_pos[k] != self.paths[k][-1]:
                self.paths[k].append(self.robots_pos[k])

    def update_stackSet(self):
        stack_set = []
        for k in range(self.n_robots):
            self.stack_set[k].pop(0)
            self.stack_set[k].append(self.state_set[k].copy())
            stack_set.append(np.array(self.stack_set[k]))
        return stack_set


    def cal_rewards(self, k):
        rewards = 0
        if self.robots_pos[k] == self.goals[k]:
            rewards += 1
        return rewards

    def cal_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def render_pathPlanning1(self):
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color obstacles
                if self.grid_global1[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                # filled color loading points
                elif self.grid_global1[j, i] in range(1, self.n_robots + 1):
                    img[j, i] = np.array(self.loading_color)
                # filled color dropOff points
                elif self.grid_global1[j, i] in range(21, self.n_robots + 21):
                    img[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    img[j, i] = np.array(self.floor_color)
                # filled color robot's positions
                for k in range(self.n_robots):
                    if self.grid_global1[j, i] == k + 11:
                        img[j, i] = np.array(self.robots_color[k])

        img_resized = cv.resize(img, (self.gridW * self.img_ratio, self.gridH * self.img_ratio),
                                interpolation=cv.INTER_AREA)

        # draw path
        for k in range(self.n_robots):
            for i, [x, y] in enumerate(self.paths[k]):
                x_point = x * self.img_ratio + self.img_ratio // 2
                y_point = y * self.img_ratio + self.img_ratio // 2
                cv.circle(img_resized, (x_point, y_point), 4, self.paths_color[k], thickness=-1)
                if i > 0:
                    cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.paths_color[k], thickness=2)
                x_prev = x_point
                y_prev = y_point

        img_croped = img_resized[self.span_y * self.img_ratio: (self.gridH - self.span_y) * self.img_ratio,
                    self.span_x * self.img_ratio: (self.gridW - self.span_x) * self.img_ratio]

        img_resized = cv.resize(img_croped, (img_croped.shape[1] // 2, img_croped.shape[0] // 2))
        cv.imshow('Mobile Robot Path Planning I', img_resized)
        cv.waitKey(1)

        return img_resized

if __name__ == "__main__":
    env = CustomEnv(2)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Randomly sample an action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()