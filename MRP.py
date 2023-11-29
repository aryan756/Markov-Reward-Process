import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from tqdm import tqdm
# Using node to represent the states and the connection between each pair
class Node:
    def __init__(self, val: str):
        self.value = val
        self.right = None
        self.left = None
        self.r_reward = 0  # the reward of stepping right
        self.l_reward = 0  # the reward of stepping left
 
    def __eq__(self, other_val) -> bool:
        return self.value == other_val
 
    def __repr__(self) -> str:
        return f"Node {self.value}"
 
# Build the Random Walk environment
class RandomWalk:
    def __init__(self):
        self.state_space = ["A", "B", "C", "D", "E"]
        # We need to make the mapping start from 1 and reserve 0 for the terminal state
        self.state_idx_map = {
            letter: idx + 1 for idx, letter in enumerate(self.state_space)
        }
        self.initial_state = "C"
        self.initial_idx = self.state_idx_map[self.initial_state]
        # Build environment as a linked list
        self.nodes = self.build_env()
        self.reset()
 
    def step(self, action: int) -> tuple:
        assert action in [0, 1], "Action should be 0 or 1"
 
        if action == 0:
            reward = self.state.l_reward
            next_state = self.state_idx_map[self.state.value] - 1
            self.state = self.state.left
        else:
            reward = self.state.r_reward
            next_state = self.state_idx_map[self.state.value] + 1
            self.state = self.state.right
 
        terminated = False if self.state else True
        return next_state, reward, terminated
 
    # reset the state to the initial node
    def reset(self):
        self.state = self.nodes
        while self.state.value != self.initial_state:
            self.state = self.state.right
 
    # building the random walk environment as a linked list
    def build_env(self) -> Node:
        values = self.state_space
        head = Node(values[0])
        builder = head
        prev = None
        for i, val in enumerate(values):
            next_node = None if i == len(values) - 1 else Node(values[i + 1])
            builder.r_reward = 0  # Set r_reward to 0 for all states
            builder.l_reward = 0  # Set l_reward to 0 for all states
            if not next_node:
                builder.r_reward = 1  # Set r_reward to 1 for state E (terminal state)
 
            builder.left = prev
            builder.right = next_node
            prev = builder
            builder = next_node
        return head
 
# TD algorithm
def TD_evaluation(alpha: float = 0.1, num_episodes=100) -> np.ndarray:
    env = RandomWalk()
    total_states = len(env.state_space) + 2  # also include terminal states on both ends
    # Initialize values
    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] += 0.5  # initial values of non-terminal state to 0.5
    gamma = 1.0
 
    V_history = np.zeros(shape=(num_episodes, total_states))
 
    # Run TD algorithm
    for episode in range(num_episodes):
        env.reset()
        state = env.initial_idx
        terminated = False
        while not terminated:
            action = np.random.choice(2)  # Random policy
            next_state, reward, terminated = env.step(action)
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
        V_history[episode] = np.copy(V)
    return V_history
 
# Rooted Mean Square Error
def rms(V_hist: np.ndarray, true_value: np.ndarray) -> np.ndarray:
    if len(true_value.shape) != 3:
        true_value = true_value.reshape(1, 1, -1)
    squared_error = (V_hist - true_value) ** 2
    rooted_mse = np.sqrt(squared_error.mean(axis=-1)).mean(axis=0)
    return rooted_mse
 
# Run TD evaluation for given episodes
def parameter_sweep(
    algorithm: Any,
    alpha_list: list,
    num_episodes: int,
    num_runs: int,
    true_value: np.ndarray,
) -> list:
    error_hist = []
    V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
    for alpha in alpha_list:
        for i in tqdm(range(num_runs), desc=f"Alpha={alpha}"):
            v_single = algorithm(alpha=alpha, num_episodes=num_episodes)
            V_hist[i] = v_single[:, 1:-1]
 
        error = rms(V_hist, true_value)
        error_hist.append(error)
    return error_hist
 
# Run the TD algorithm and plot out the errors
def algorithm_comparison(
    num_episodes: int, num_runs: int, true_value: np.ndarray
) -> None:
    alpha_list_td = [0.05, 0.1, 0.15]
    td_error_hist = parameter_sweep(
        TD_evaluation, alpha_list_td, num_episodes, num_runs, true_value
    )
 
    # Plotting the result
    font_dict = {"fontsize": 11}
    colors = ["mediumseagreen", "steelblue", "orchid"]
 
    plt.figure(figsize=(9, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
 
    plt.xlabel("Walks/Episodes", fontdict=font_dict)
    plt.ylabel("RMS error", fontdict=font_dict)
 
    # plot TD errors with different line weights
    color = colors[1]
    linewidths = [1.8, 1.3, 0.9]
    for i, error in enumerate(td_error_hist):
        plt.plot(
            error,
            c=color,
            linewidth=linewidths[i],
            label=f"TD $\\alpha$={alpha_list_td[i]}",
        )
 
    plt.title(
        "Empirical RMS error, averaged over states and 100 runs",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend()
    plt.show()
 
def main():
    #b = Bandit()
    #b.run_experiments()

    # The true value of the random policy for RW is provided as follows
    true_value = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    num_runs = 100
    num_episodes = 100

    # Example b: RMS error over different setups
    
    algorithm_comparison(num_episodes, num_runs, true_value)


if __name__=='__main__':

    main()