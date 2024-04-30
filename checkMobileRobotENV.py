from myEnv import CustomMobileRobotEnv
from stable_baselines3.common.env_checker import check_env
import pygame

# Your CustomEnv class definition and other code here...

if __name__ == "__main__":
    pygame.init()
    # screen = pygame.display.set_mode((800, 600))
    env = CustomMobileRobotEnv(2)

    # Check the environment with a seed
    check_env(env)
