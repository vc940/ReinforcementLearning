import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")

model = DQN(
    "MlpPolicy",  
    env,
    learning_rate=0.001,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)

model.learn(total_timesteps=600_000)

model.save("dqn_lunarlander")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

env.close()
