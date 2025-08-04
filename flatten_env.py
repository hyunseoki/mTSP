import gym
import numpy as np
from gym import spaces

class FlattenedEnv(gym.Env):
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        self.num_agents = base_env.num_agents
        self.num_nodes = base_env.num_nodes  # 수정: 그대로 사용

        # Flatten된 observation: 각 agent 위치 + visited (0 or 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.num_agents + self.num_nodes,),
            dtype=np.float32
        )

        # Action: (agent_id * num_task_nodes + task_node_id)
        self.action_space = spaces.Discrete(self.num_agents * self.num_nodes)

    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)

    def _flatten_obs(self, obs):
        task_visited = obs['visited'][self.num_agents:]
        total_nodes = self.env.num_agents + self.env.num_nodes
        current_norm = obs['current_nodes'] / total_nodes  # 수정
        visited_f = task_visited.astype(np.float32)
        return np.concatenate([current_norm, visited_f])

    def step(self, action):
        agent_id = action // self.num_nodes
        task_node_id = action % self.num_nodes
        actual_node = self.num_agents + task_node_id

        obs, reward, done, info = self.env.step((agent_id, actual_node))
        return self._flatten_obs(obs), reward, done, info

    def get_valid_actions(self):
        mask = self.env.get_action_mask()
        valid_actions = []
        for agent_id in range(self.num_agents):
            for task_node_id in range(self.num_nodes):
                actual_node = self.num_agents + task_node_id
                if mask[agent_id, actual_node] == 1:
                    action = agent_id * self.num_nodes + task_node_id
                    valid_actions.append(action)
        return valid_actions

    def render(self, mode='human'):
        self.env.render()


if __name__ == '__main__':
    from env import mTSPEnv
    base_env = mTSPEnv(num_agents=2, num_nodes=4)
    env = FlattenedEnv(base_env)

    obs = env.reset()
    done = False

    print("Initial observation:", obs)

    step_count = 0
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("No valid actions left!")
            break

        action = np.random.choice(valid_actions)
        obs, reward, done, info = env.step(action)

        print(f"\nStep {step_count}")
        print(f"Action taken (flattened): {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}, Done: {done}")
        step_count += 1

    print("\nFinal Render:")
    env.render()
