import gymnasium as gym
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("CartPole(scratch)/dqn-1000.h5", compile=False)
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset()[0]

done = False
total_reward = 0

while not done:
    action_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    action = np.argmax(action_values) 
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()
