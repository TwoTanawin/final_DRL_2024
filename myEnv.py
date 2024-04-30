import numpy as np
import math
import cv2 as cv
import random
import gymnasium as gym
from gymnasium import spaces


class MobileRobot(gym.Env):

    metadata = {"render.modes": ["human"], "render.fps": 30}

    def __init__(self, n_robots):
        # Parameters for path planning
        # Define action and observation space
        self.action_space = spaces.Discrete(8)  # 8 discrete actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(8, 172, 300), dtype=np.uint8)

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

        self.reset()


    def reset(self, seed=None, options=None):
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

        # Concatenate the first 3 stacks to match the shape (3, 172, 300)
        obs = np.concatenate(stack_set[:3], axis=0)

        obs = np.uint8(obs)
        info = {}  # Placeholder for additional information, if needed

        return obs, info


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


    def step(self, actions):
        new_positions = []
        rewards = []
        terminals = []
        infos = []

        for k in range(self.n_robots):
            new_position = self.robots_pos[k].copy()

            # Handle each action separately for each robot
            print(actions)
            action = actions
            if action == 0: # move up-left
                new_position[0] -= 1
                new_position[1] -= 1
            elif action == 1: # move up
                new_position[1] -= 1
            elif action == 2: # move up-right
                new_position[0] += 1
                new_position[1] -= 1
            elif action == 3: # move left
                new_position[0] -= 1
            elif action == 4: # move right
                new_position[0] += 1
            elif action == 5: # move down-left
                new_position[0] -= 1
                new_position[1] += 1
            elif action == 6: # move down
                new_position[1] += 1
            elif action == 7: # move down-right
                new_position[0] += 1
                new_position[1] += 1

            dist = self.cal_distance(self.robots_pos[k], self.goals[k]) - self.cal_distance(new_position, self.goals[k])
            line = self.cal_distance(new_position, self.robots_pos[k])
            reward = dist - line
            terminal = False
            info = {}

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

        # Concatenate the first 3 stacks to match the shape (3, 172, 300)
        obs = np.concatenate(stack_set[:3], axis=0)
        obs = np.uint8(obs)
        
        truncated = False

        print(rewards)

        return obs, reward, terminal, truncated, info



    def reachGoal(self, pos, k):
        if pos == self.goals[k]:
            return True
        return False

    def HitObstacle(self, pos, grid):
        if grid[pos[1], pos[0]] == -1:
            return True
        return False

    def cal_distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def update_gridSet(self, new_positions):
        for k in range(self.n_robots):
            if self.grid_global1[new_positions[k][1], new_positions[k][0]] != 0:
                self.grid_global1[new_positions[k][1], new_positions[k][0]] = k + 1
                self.grid_global1[self.robots_pos[k][1], self.robots_pos[k][0]] = 0
            else:
                self.grid_global1[new_positions[k][1], new_positions[k][0]] = k + 1
            self.robots_pos[k] = new_positions[k].copy()

    def add_paths(self):
        for k in range(self.n_robots):
            if self.robots_pos[k] != self.paths[k][-1]:
                self.paths[k].append(self.robots_pos[k])

    def update_stackSet(self):
        stack_set = []
        for k in range(self.n_robots):
            self.stack_set[k].pop(0)
            self.stack_set[k].append(self.state_set[k])
            stack_set.append(np.array(self.stack_set[k]))
        return stack_set
    
    def render(self, mode='human'):
        img = np.zeros((self.gridH, self.gridW, 3), dtype=np.uint8)
        for i in range(self.gridW):
            for j in range(self.gridH):
                # filled color obstacles
                if self.grid_global1[j, i] == -1:
                    img[j, i] = np.array(self.obstacle_color)
                # filled color loading points
                elif self.grid_global1[j, i] in range(1, self.n_robots+1):
                    img[j, i] = np.array(self.loading_color)
                # filled color dropOff points
                elif self.grid_global1[j, i] in range(21, self.n_robots+21):
                    img[j, i] = np.array(self.dropOff_color)
                # filled color floor
                else:
                    img[j, i] = np.array(self.floor_color)
                # filled color robot's positions
                for k in range(self.n_robots):
                    if self.grid_global1[j, i] == k + 11:
                        img[j, i] = np.array(self.robots_color[k])
                
        img_resized = cv.resize(img, (self.gridW*self.img_ratio, self.gridH*self.img_ratio), interpolation=cv.INTER_AREA)

        # draw path
        for k in range(self.n_robots):
            for i, [x,y] in enumerate(self.paths[k]):
                x_point = x*self.img_ratio + self.img_ratio//2
                y_point = y*self.img_ratio + self.img_ratio//2
                cv.circle(img_resized, (x_point, y_point), 4, self.paths_color[k], thickness = -1)
                if i > 0:
                    cv.line(img_resized, (x_prev, y_prev), (x_point, y_point), self.paths_color[k], thickness = 2)
                x_prev = x_point
                y_prev = y_point

        img_croped = img_resized[self.span_y*self.img_ratio: (self.gridH - self.span_y)*self.img_ratio,
                                self.span_x*self.img_ratio: (self.gridW - self.span_x)*self.img_ratio]

        img_resized = cv.resize(img_croped, (img_croped.shape[1]//2, img_croped.shape[0]//2))

        # cv.imshow('Moibile Robot Path Planning I', img_resized)

        # cv.waitKey(1)

        # return img_resized
        if mode == 'rgb_array':
            return img_resized  # Return the image array directly
        elif mode == 'human':
            cv.imshow('Mobile Robot Path Planning', img_resized)
            cv.waitKey(1)
        else:
            super(MobileRobot, self).render(mode=mode)  # Call superclass render method with the specified mode

# env = CustomEnv(n_robots=3)
# state = env.reset()
# print(state)
    def close(self):
        cv.destroyAllWindows()
