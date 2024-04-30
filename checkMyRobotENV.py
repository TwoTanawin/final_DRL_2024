from myEnv2 import CustomEnv
from stable_baselines3.common.env_checker import check_env
import pygame

# Your CustomEnv class definition and other code here...

# if __name__ == "__main__":
#     pygame.init()
    # # screen = pygame.display.set_mode((800, 600))
    # env = MobileRobot(2)

    # # Check the environment with a seed
    # check_env(env)
if __name__ == "__main__":
    pygame.init()
    # screen = pygame.display.set_mode((800, 600))
    env = CustomEnv()

    # Create a list of actions for each robot
    # actions = [0, 1]  # Replace 1 and 2 with the desired actions for each robot

    # Check the environment with a seed
    check_env(env)