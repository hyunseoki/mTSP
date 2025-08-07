import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from flatten_env import FlattenedEnv
from mtsp_env import mTSPEnv  # base environment
import matplotlib.pyplot as plt


class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=10000):
        self.q_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = deque(maxlen=buffer_size)


    def select_action(self, state, valid_actions, epsilon):
        if random.random() < epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state))
        q_valid = {a: q_values[a].item() for a in valid_actions}
        return max(q_valid, key=q_valid.get)


    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))


    def train_step(self):
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s_next, d = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_next = torch.FloatTensor(s_next)
        d = torch.FloatTensor(d).unsqueeze(1)

        q = self.q_net(s).gather(1, a)
        q_next = self.target_net(s_next).max(1)[0].unsqueeze(1)
        q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def evaluate(env, q_net):
    state = env.reset()
    state = np.concatenate([state, env.distance_matrix_norm])
    done = False
    total_reward = 0
    while not done:
        valid_actions = env.get_valid_actions()
        with torch.no_grad():
            q_values = q_net(torch.tensor(state, dtype=torch.float32))
            # invalid actions에 대해 -inf 처리
            q_values[[i for i in range(len(q_values)) if i not in valid_actions]] = -float('inf')
            action = torch.argmax(q_values).item()
        state, reward, done, _ = env.step(action)
        state = np.concatenate([state, env.distance_matrix_norm])
        total_reward += reward

    env.render()
    print(f"Eval Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    base_env = mTSPEnv(num_agents=3, num_task_nodes=8)
    env = FlattenedEnv(base_env)

    state_dim = env.observation_space.shape[0] + env.distance_matrix_norm.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 2000
    episode_rewards = []  # <-- 추가: reward 기록용

    for episode in range(num_episodes):
        state = env.reset()
        state = np.concatenate([state, env.distance_matrix_norm])
        done = False
        total_reward = 0
        epsilon = max(0.05, 1.0 - episode / num_episodes)

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, epsilon)
            next_state, reward, done, _ = env.step(action)

            agent.store(state, action, reward, np.concatenate([next_state, env.distance_matrix_norm]), done)
            agent.train_step()

            state = np.concatenate([next_state, env.distance_matrix_norm])
            total_reward += reward

        episode_rewards.append(total_reward)  # <-- 추가: total_reward 기록

        if episode % 10 == 0:
            agent.update_target()
            print(f"Episode {episode}: total reward = {total_reward:.2f}")

    # 학습 종료 후 evaluate 실행
    evaluate(env, agent.q_net)

    # <-- 추가: reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode-wise Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







