import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make("LunarLander-v3",render_mode = 'human')

# Load the trained model
model = DQN.load("Lunar_lander(stable_baseline)/dqn_lunarlander.zip")

# Evaluate the loaded model
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  # Visualize the environment
    action, _ = model.predict(obs)  # Get the action from the loaded model
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Total Reward: {total_reward}")
env.close()
