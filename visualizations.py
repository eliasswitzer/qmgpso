import matplotlib.pyplot as plt

def plot_fitness(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['best_fitness'], label="Best Fitness", color='blue', linewidth=2)
    plt.plot(history['avg_fitness'], label="Average Swarm Fitness", color='orange', linestyle='--')
    plt.title("PSO Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diversity(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['diversity'], color='green')
    plt.title("Swarm Diversity Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Swarm Diversity (Mean Distance to Global Best)")
    plt.grid(True)
    plt.show()