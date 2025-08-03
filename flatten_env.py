import gym
import numpy as np
from gym import spaces

class FlattenedEnv(gym.Env):
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        self.num_agents = base_env.num_agents
        self.num_task_nodes = base_env.num_nodes - self.num_agents

        # Flatten된 observation: 각 agent 위치 + visited (0 or 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.num_agents + self.num_task_nodes,),
            dtype=np.float32
        )

        # Action: (agent_id * num_task_nodes + task_node_id)
        self.action_space = spaces.Discrete(self.num_agents * self.num_task_nodes)

    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)

    def _flatten_obs(self, obs):
        # current_nodes: [0, 1]
        # visited: [1, 0, 1, 0]  # 전체 노드 기준, 0,1은 시작노드 → 제외
        task_visited = obs['visited'][self.num_agents:]
        current_norm = obs['current_nodes'] / self.env.num_nodes
        visited_f = task_visited.astype(np.float32)
        return np.concatenate([current_norm, visited_f])

    def step(self, action):
        agent_id = action // self.num_task_nodes
        task_node_id = action % self.num_task_nodes
        # 실제 노드 번호 (0,1: 시작노드 → 2부터 시작)
        actual_node = self.num_agents + task_node_id

        obs, reward, done, info = self.env.step((agent_id, actual_node))
        return self._flatten_obs(obs), reward, done, info

    def render(self, mode='human'):
        self.env.render()
