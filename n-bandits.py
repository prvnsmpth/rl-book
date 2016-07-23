import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    
    def __init__(self, n):
        self.num_levers = n
        self.lever_rewards = np.random.normal(0, 1, n)

    def get_reward(self, i):
        return self.lever_rewards[i-1] + np.random.normal(0, 1)

class Agent:

    def __init__(self, eps, bandit):
        self.eps = eps
        self.bandit = bandit
        self.reward_est = [(0, 0) for i in range(bandit.num_levers)]

    def pick_lever(self):
        lever, _ = max(enumerate(self.reward_est), key=lambda e: e[1][1])
        if self.eps > 0:
            explore = np.random.rand()
            if explore <= self.eps:
                # Explore!
                lever = np.random.randint(0, self.bandit.num_levers)
        current_est = self.reward_est[lever]
        reward = self.bandit.get_reward(lever) 
        self.reward_est[lever] = (current_est[0]+1, 
                (current_est[0]*current_est[1]+reward)/(current_est[0]+1))
        return reward

def run(agents, num_bandits, steps):
    """Returns the average reward received by the agents on each step"""
    return [
        np.mean([eps_001_agents[i].pick_lever() 
                    for i in range(num_bandits)])
        for step in range(steps)]

steps = 1000
num_bandits = 2000

bandits = [Bandit(10) for i in range(num_bandits)]

greedy_agents = [Agent(0, bandits[i]) for i in range(num_bandits)]
eps_001_agents = [Agent(0.01, bandits[i]) for i in range(num_bandits)]
eps_01_agents = [Agent(0.1, bandits[i]) for i in range(num_bandits)]

greedy_avg_rewards = run(greedy_agents, num_bandits, steps)
eps_001_rewards = run(eps_001_agents, num_bandits, steps)
eps_01_rewards = run(eps_01_agents, num_bandits, steps)

plt.plot(range(steps), greedy_avg_rewards, 
        range(steps), eps_001_rewards,
        range(steps), eps_01_rewards)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.show()
