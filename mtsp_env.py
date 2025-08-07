import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


# class mSTP():
#     def __init__(self, num_agents=2, num_nodes=4, size=100):
#         assert num_agents < num_nodes
#         self.num_nodes = num_nodes
#         self.num_agents = num_agents
#         self.size = size
    
#     def make_problem(self):
#         coords = np.random.rand(self.num_agents + self.num_nodes, 2) * self.size
#         dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
#         return coords, dist_matrix


class mTSPEnv(gym.Env):
    def __init__(self, num_agents=2, num_task_nodes=4, size=100):
        super().__init__()
        assert num_agents < num_task_nodes
        self.num_task_nodes = num_task_nodes
        self.num_agents = num_agents
        self.num_total_nodes = num_agents + num_task_nodes
        self.size = size

        # action: agent 선택 후, 다음 방문 노드 선택 → (agent_id, node_id)
        self.action_space = spaces.MultiDiscrete([self.num_agents, self.num_total_nodes])
        # observation: 각 agent의 current node + 전체 방문 여부
        self.observation_space = spaces.Dict({
            # 각 agent의 현재 위치를 나타냄. 예를 들어 agent가 2명이고 노드가 4개면 [0, 1]처럼 표현됨.
            # 각 값은 해당 agent가 위치한 노드 번호(0 ~ num_nodes-1)
            'current_nodes': spaces.MultiDiscrete([self.num_total_nodes] * self.num_agents),
            # 각 노드의 방문 여부를 0 또는 1로 나타냄. 예: [1, 0, 1, 0]이면 0번, 2번 노드는 방문, 1번, 3번은 미방문
            'visited': spaces.MultiBinary(self.num_total_nodes),
        })

        self.reset()


    def get_action_mask(self):
        """
        각 agent별로 방문 가능한 노드만 1, 나머지는 0인 마스크 반환.
        shape: (num_agents, num_nodes)
        """
        mask = np.zeros((self.num_agents, self.num_total_nodes), dtype=np.int32)
        for agent_id in range(self.num_agents):
            for node_id in range(self.num_total_nodes):
                if not self.visited[node_id]:
                    mask[agent_id, node_id] = 1
        return mask


    def _make_problem(self):
        coords = np.random.rand(self.num_total_nodes, 2) * self.size
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return coords, dist_matrix


    def reset(self):
        # 초기 상태
        self.coords, self.distance_matrix = self._make_problem()

        self.current_nodes = np.arange(self.num_agents)  # [0, 1, ..., num_agents-1] 시작위치, agent0은 0번 노드, agent1은 1번 노드 ...
        self.visited = np.zeros(self.num_total_nodes, dtype=bool)
        self.visited[self.current_nodes] = True
        self.agent_paths = [[start] for start in self.current_nodes]
        self.agent_costs = np.zeros(self.num_agents)
        return self._get_obs()


    def _get_obs(self):
        return {
            'current_nodes': self.current_nodes.copy(),
            'visited': self.visited.copy()
        }

    def step(self, action):
        agent_id, next_node = action
        cur_node = self.current_nodes[agent_id]

        # 방어코드
        if self.visited[next_node]:
            reward = -100  # 페널티: 이미 방문한 노드
            done = False
            return self._get_obs(), reward, done, {'error': 'Node already visited'}

        # 이동
        cost = self.distance_matrix[cur_node, next_node]
        self.agent_costs[agent_id] += cost
        self.current_nodes[agent_id] = next_node
        self.visited[next_node] = True
        self.agent_paths[agent_id].append(next_node)
        done = self.visited.all()

        # if done:
        #     reward = -max(self.agent_costs) * 100
        # else:
        #     reward = -self.distance_matrix[cur_node, next_node]

        if done:
            agent_total_distances = self.agent_costs  # shape: (num_agents,)
            max_dist = max(agent_total_distances)
            reward = -max_dist
        else:
            reward = 0
            
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        for i in range(self.num_agents):
            print(f"Agent {i}: Path = {self.agent_paths[i]}, Cost = {self.agent_costs[i]}")
        print(f"Total max cost: {np.max(self.agent_costs)}")

        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        plt.figure(figsize=(8, 6))

        n_start = self.num_agents  # 출발점 개수 (보통 2개: 0, 1)
        total_nodes = self.num_agents + self.num_task_nodes

        # Plot 출발점 (0 ~ num_agents-1): 빨간 점 + "start"
        plt.scatter(self.coords[:n_start, 0], self.coords[:n_start, 1],
                    c='red', s=100, zorder=5, label='Start nodes')

        for idx, (x, y) in enumerate(self.coords[:n_start]):
            plt.text(x + 1, y + 1, "start", fontsize=10, color='red')

        # Plot task nodes: 검정 점 + 1부터 시작하는 번호
        task_coords = self.coords[n_start:]
        plt.scatter(task_coords[:, 0], task_coords[:, 1], c='black', zorder=5)

        for i, (x, y) in enumerate(task_coords):
            plt.text(x + 1, y + 1, str(i + 1), fontsize=10)

        # 각 agent 경로 시각화
        for agent_id, path in enumerate(self.agent_paths):
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                plt.plot([self.coords[a, 0], self.coords[b, 0]],
                        [self.coords[a, 1], self.coords[b, 1]],
                        color=colors[agent_id % len(colors)],
                        linewidth=2,
                        label=f'Agent {agent_id} (Cost: {self.agent_costs[agent_id]:.1f})' if i == 0 else None)

        plt.title(f"mTSP Routes, Maximum Cost : {max(self.agent_costs):.1f}")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    env = mTSPEnv(num_agents=4, num_task_nodes=7)
    obs = env.reset()

    done = False
    while not done:
        # 임의 정책: 방문 안 한 노드 중 하나 선택
        for agent_id in range(env.num_agents):
            possible_nodes = np.where(~obs['visited'])[0]
            if len(possible_nodes) == 0:
                continue
            action = (agent_id, np.random.choice(possible_nodes))
            obs, reward, done, _ = env.step(action)
            if done:
                break

    env.render()
