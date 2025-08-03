import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train_dqn(env, num_episodes=500, gamma=0.99, lr=1e-3, batch_size=64, 
              buffer_size=10000, target_update=10, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500):

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(1000):  # max steps
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 학습
            if len(buffer) >= batch_size:
                s, a, r, s_, d = buffer.sample(batch_size)

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_vals = target_net(s_).max(1)[0]
                    target = r + gamma * (1 - d) * next_q_vals

                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # target network update
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if episode % 10 == 0:
            print(f"Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return q_net


def evaluate(env, q_net):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            action = q_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.render()
    print(f"Eval Total Reward: {total_reward:.2f}")


if __name__ == '__main__':
    from env import mTSPEnv
    from flatten_env import FlattenedEnv
    base_env = mTSPEnv(num_agents=2, num_nodes=6)
    env = FlattenedEnv(base_env)
    q_net = train_dqn(env, num_episodes=500)
    evaluate(env, q_net)
