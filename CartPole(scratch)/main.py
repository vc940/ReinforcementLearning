import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense

env = gym.make("CartPole-v1")
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(2, activation='linear'),  # Linear output for Q-values
])
target_model = clone_model(model)
target_model.set_weights(model.get_weights())

states, next_states, actions, dones, rewards = [], [], [], [], []
epsilon = 1.0
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Policy function
def policy(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(tf.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values)

# Reward function
def reward_fn(state, final_state):
    cart_pos = state[0]
    velocity = state[1]
    pole_angle = state[2]
    angular_vel = state[3]
    velocity_f = final_state[1]
    pole_angle_f = final_state[2]
    angular_vel_f = final_state[3]
    cart_pos_f = final_state[0]
    initial_pos = pole_angle**2 + velocity**2 + angular_vel**2 + cart_pos**2
    final_pos = pole_angle_f**2 + velocity_f**2 + angular_vel_f**2 + cart_pos_f**2
    return final_pos - initial_pos

# Training loop
for i in range(10000):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(state)
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = env_reward
        if len(states) > 10000:
            states.pop(0)
            next_states.pop(0)
            actions.pop(0)
            dones.pop(0)
            rewards.pop(0)
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        dones.append(done)
        rewards.append(reward)

        # Training step
        if i > 15:
            states_tensor = tf.convert_to_tensor(states)
            next_states_tensor = tf.convert_to_tensor(next_states)
            actions_tensor = tf.convert_to_tensor(actions)
            dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
            rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)

            with tf.GradientTape() as tape:
                random_index = tf.random.uniform(shape=(32,), minval=0, maxval=len(states), dtype=tf.int32)
                random_state = tf.gather(states_tensor, random_index)
                random_next_state = tf.gather(next_states_tensor, random_index)
                random_action = tf.gather(actions_tensor, random_index)
                random_done = tf.gather(dones_tensor, random_index)
                random_reward = tf.gather(rewards_tensor, random_index)

                target = random_reward + 0.9 * (1. - random_done) * tf.reduce_max(target_model(random_next_state), axis=1)
                predicted_q_values = tf.reduce_sum(
                    model(random_state) * tf.one_hot(random_action, depth=2), axis=1
                )
                loss = tf.keras.losses.MeanSquaredError()(target, predicted_q_values)

            grads = tape.gradient(loss, model.trainable_variables)
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_reward += env_reward
        state = next_state

    if i % 10 == 0:
        epsilon = max(0.1, epsilon * 0.995)
        target_model.set_weights(model.get_weights())
    if i % 1000 ==0:
        model.compile(loss ='mse',optimizer = optimizer)
        model.save(f'dqn-{i}.h5')
    print(f"Episode {i + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env.close()
model.compile(loss ='mse',optimizer = optimizer)
model.save('dqn.h5')
