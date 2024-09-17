"""
Reproduction of "The Evolution of Ethnocentrism" by Ross A. Hammond and Robert Axelrod (2006)
"""

import random

import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self, tag, strat_in, strat_out, base_ptr=0.12):
        self.tag = tag  # Tag specifying group color
        self.strat_in = strat_in  # Cooperation strategy with in-group
        self.strat_out = strat_out  # Cooperation strategy with out-group
        self.ptr = base_ptr  # Potential to Reproduce (PTR)

    def mutate(self, mutation_rate=0.005):
        """Randomly mutate the agent's traits with a given mutation rate."""
        if random.random() < mutation_rate:
            self.strat_in = 1 - self.strat_in  # Mutate strategy
        if random.random() < mutation_rate:
            self.strat_out = 1 - self.strat_out  # Mutate strategy


class Lattice:
    def __init__(
        self,
        size=50,
        base_ptr=0.12,
        mutation_rate=0.005,
        cost=0.01,
        benefit=0.03,
        death_rate=0.10,
        colors=4,
        immigration_rate=1,
        symmetric_interaction=True,
    ):
        self.size = size  # Size of the lattice (50x50)
        self.grid = np.zeros((size, size), dtype=object)
        self.base_ptr = base_ptr
        self.mutation_rate = mutation_rate
        self.cost = cost
        self.benefit = benefit
        self.death_rate = death_rate
        self.colors = colors
        self.immigration_rate = immigration_rate
        self.total_agents = 0
        self.ethnocentric_count = 0  # Count for agents using ethnocentric strategy
        self.cooperative_count = 0  # Count for cooperative actions
        self.symmetric_interaction = (
            symmetric_interaction  # New parameter to toggle interaction type
        )

    def add_agent(self, x, y):
        """Adds an agent with random traits at a given position."""
        tag = random.choice(list(range(self.colors)))
        strat_in = random.choice([0, 1])  # Strategy towards in-group (cooperate or defect)
        strat_out = random.choice([0, 1])  # Strategy towards out-group (cooperate or defect)
        self.grid[x, y] = Agent(tag, strat_in, strat_out, self.base_ptr)

        # Check if agent has the ethnocentric strategy (cooperate in-group, defect out-group)
        if strat_in == 1 and strat_out == 0:
            self.ethnocentric_count += 1

    def wrap_around(self, x, y):
        """Handles toroidal wrapping of lattice edges."""
        return x % self.size, y % self.size

    def get_neighbors(self, x, y):
        """Returns the neighbors of the agent at position (x, y) in Von Neumann geometry."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = self.wrap_around(x + dx, y + dy)
            if self.grid[nx, ny] != 0:
                neighbors.append((nx, ny))
        return neighbors

    def immigrate(self, num_immigrants):
        """Simulate the immigration of new agents."""
        for _ in range(num_immigrants):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x][y] == 0:  # Place in an empty spot
                new_agent = Agent.random_agent()  # Create a random new agent
                self.grid[x][y] = new_agent
                self.update_counts_after_immigration(new_agent)

    def is_ethnocentric(self, agent):
        """Determine if an agent is ethnocentric (cooperates with in-group, defects with out-group)."""
        return agent.strat_in == 1 and agent.strat_out == 0

    def interact(self):
        """Simulate interaction between agents, either symmetric or unidirectional based on the parameter."""
        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if agent == 0:
                    continue

                neighbors = self.get_neighbors(x, y)
                for nx, ny in neighbors:
                    neighbor = self.grid[nx][ny]
                    if neighbor == 0:
                        continue

                    # Agent A (focal agent) decides whether to help neighbor (N)
                    if agent.tag == neighbor.tag:  # In-group interaction
                        if agent.strat_in == 1:  # A helps N (same-tag)
                            agent.ptr -= self.cost
                            neighbor.ptr += self.benefit
                            self.cooperative_count += 1  # Count cooperative action
                    else:  # Out-group interaction
                        if agent.strat_out == 1:  # A defects from helping N
                            agent.ptr -= self.cost
                            neighbor.ptr += self.benefit
                        else:  # A helps N (out-group)
                            self.cooperative_count += 1  # Count cooperative action

                    # Symmetric interaction if enabled
                    if self.symmetric_interaction:
                        if neighbor.tag == agent.tag:  # In-group interaction
                            if neighbor.strat_in == 1:  # N helps A (same-tag)
                                neighbor.ptr -= self.cost
                                agent.ptr += self.benefit
                                self.cooperative_count += 1  # Count cooperative action
                        else:  # Out-group interaction
                            if neighbor.strat_out == 1:  # N defects from helping A
                                neighbor.ptr -= self.cost
                                agent.ptr += self.benefit
                            else:  # N helps A (out-group)
                                self.cooperative_count += 1  # Count cooperative action

    def update_counts_after_immigration(self, agent):
        """Update total and ethnocentric counts after adding a new agent."""
        self.total_agents += 1
        if self.is_ethnocentric(agent):
            self.ethnocentric_count += 1

    def update_counts_after_death(self, agent):
        """Update total and ethnocentric counts after removing an agent."""
        self.total_agents -= 1
        if self.is_ethnocentric(agent):
            self.ethnocentric_count -= 1

    def reproduce(self):
        """Simulate reproduction based on agents' PTR and mutation."""
        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if agent == 0:
                    continue
                if random.random() < agent.ptr:  # Reproduction chance based on PTR
                    neighbors = self.get_neighbors(x, y)
                    empty_spots = [(nx, ny) for nx, ny in neighbors if self.grid[nx][ny] == 0]
                    if empty_spots:
                        nx, ny = random.choice(empty_spots)
                        new_agent = agent.clone_with_mutation(self.mutation_rate)
                        self.grid[nx][ny] = new_agent
                        self.update_counts_after_immigration(new_agent)

    def death(self):
        """Simulate death of agents with a given probability."""
        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if agent == 0:
                    continue
                if random.random() < self.death_rate:
                    self.update_counts_after_death(agent)
                    self.grid[x][y] = 0  # Remove agent from the grid

    def get_ethnocentric_percentage(self):
        """Calculate the percentage of ethnocentric agents."""
        if self.total_agents == 0:
            return 0
        return (self.ethnocentric_count / self.total_agents) * 100

    def step(self):
        """Advance the simulation by one time step."""
        self.immigrate(1)  # 1 immigrant per period
        self.interact()
        self.reproduce()
        self.death()

    def visualize(self):
        """Visualize the current state of the lattice."""
        # Create a grid of tags (or None for empty cells)
        grid_visual = np.zeros((self.size, self.size))

        for x in range(self.size):
            for y in range(self.size):
                agent = self.grid[x][y]
                if agent != 0:
                    grid_visual[x][y] = (
                        agent.tag + 1
                    )  # To differentiate empty (0) from agents (1-4)

        # Create the plot
        plt.imshow(grid_visual, cmap="tab10", origin="upper")
        plt.title("Lattice at time step")
        plt.colorbar(ticks=[0, 1, 2, 3, 4], label="Group Tags")
        plt.show()

    def run(self, run_length=2000):
        """Run the simulation and record the ethnocentric and cooperative data."""
        ethnocentric_counts = []
        cooperative_counts = []

        for time_step in range(run_length):
            self.step()

            # Record counts for the last 100 time steps
            if time_step >= run_length - 100:
                ethnocentric_counts.append(self.ethnocentric_count)
                cooperative_counts.append(self.cooperative_count)

        # Calculate averages
        avg_ethnocentric = np.mean(ethnocentric_counts)
        avg_cooperative = np.mean(cooperative_counts)

        # Reset counts for the next run
        self.ethnocentric_count = 0
        self.cooperative_count = 0

        return avg_ethnocentric, avg_cooperative


def run_multiple_simulations(lattice_params, num_runs=10, run_length=2000):
    """Run the simulation multiple times and calculate the mean and standard error."""
    ethnocentric_results = []
    cooperative_results = []

    for _ in range(num_runs):
        lattice = Lattice(**lattice_params)
        avg_ethnocentric, avg_cooperative = lattice.run(run_length)
        ethnocentric_results.append(avg_ethnocentric)
        cooperative_results.append(avg_cooperative)

    # Compute mean and standard error
    ethnocentric_mean = np.mean(ethnocentric_results)
    cooperative_mean = np.mean(cooperative_results)
    ethnocentric_se = np.std(ethnocentric_results) / np.sqrt(num_runs)
    cooperative_se = np.std(cooperative_results) / np.sqrt(num_runs)

    return ethnocentric_mean, cooperative_mean, ethnocentric_se, cooperative_se


# Example: Run for standard parameters
standard_params = {
    "size": 50,
    "base_ptr": 0.12,
    "mutation_rate": 0.005,  # 0.5% mutation rate
    "cost": 0.01,  # 1% cost of giving help
    "benefit": 0.03,  # 3% benefit of receiving help
    "death_rate": 0.10,
    "colors": 4,
    "immigration_rate": 1,  # 1 immigrant per period
}

ethnocentric_mean, cooperative_mean, ethnocentric_se, cooperative_se = run_multiple_simulations(
    standard_params
)

print(f"Ethnocentric mean: {ethnocentric_mean:.2f} ± {ethnocentric_se:.2f}")
print(f"Cooperative mean: {cooperative_mean:.2f} ± {cooperative_se:.2f}")


lattice = Lattice(size=50)

for time_step in range(1, 2001):
    lattice.step()

    if time_step > 1500 and time_step % 100 == 0:
        print(f"Time Step: {time_step}")
        lattice.visualize()
