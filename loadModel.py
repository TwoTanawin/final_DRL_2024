from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from mainEnv_timmer import CustomEnv
import time
import json

# Initialize a dictionary to store episode times
episode_times = {}


# Create a new en3ironment for inference
env = CustomEnv()

model_path = "/home/two-asus/Documents/ait/drl/final_DRL_2024/report/A2C_MLP_mainEnv_timmer_RobotPath/model/80000.zip"

# Load the trained model
model = A2C.load(model_path, env=env)

text = model_path.split("/")

last_text = text[-1].split(".")

# Lists to store rewards and episode lengths
episode_rewards = []
episode_lengths = []

# Inference loop
episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    start_time = time.time()
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        env.render(mode='human')  # Render the environment in human mode during inference
        print(f"reward : {reward}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

    # Calculate and print time taken
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Episode {ep+1} took {time_taken:.2f} seconds")

    episode_times[f"Episode {ep+1}"] = time_taken
    print(f"Episode {ep+1} took {time_taken:.2f} seconds")

# Save episode times to a JSON file
json_file_path = "episode_times.json"
with open(json_file_path, 'w') as f:
    json.dump(episode_times, f)

# Plotting rewards
plt.plot(np.arange(1, episodes + 1), episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward Plot')
plt.grid(True)
plt.show()

plt.tight_layout()
plt.savefig(f"report/img1_{text[8]}_{last_text[0]}.jpg")
plt.show()

env.close()  # Close the environment after inference