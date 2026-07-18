import numpy as np
import matplotlib.pyplot as plt

def plot_pareto_front(archive, objective_labels=None):
    """Scatter plot of the archive's current Pareto front"""
    fitnesses = np.array(archive.fitnesses)
    if fitnesses.size == 0:
        print("Archive is empty, nothing to plot.")
        return
    
    num_objectives = fitnesses.shape[1]
    if objective_labels is None:
        objective_labels = [f"f{i+1}" for i in range(num_objectives)]
    
    if num_objectives == 2:
        fig, ax = plt.subplots(figsize=(7,6))
        ax.scatter(fitnesses[:,0], fitnesses[:,1], c='tab:blue', s=20)
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    elif num_objectives == 3:
        fig = plt.subplots(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fitnesses[:,0], fitnesses[:,1], fitnesses[:,2], c='tab:blue', s=20)
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        ax.set_zlabel(objective_labels[2])
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("Too many objectives for a plot")

def plot_archive_size(history):
    """Plots archive size over iterations"""
    plt.figure(figsize=(10,5))
    plt.plot(history['archive_size'], color='purple')
    plt.title("Archive Size Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Non-Dominated Solutions")
    plt.grid(True)
    plt.show()

def plot_pareto_front_history(history, objective_labels=None, num_snapshots=6):
    """Overlays several archive snapshots to show how the Pareto front changed over time from the changing environment"""
    snapshots = history['archive_snapshots']
    if not snapshots:
        print("No archive snapshots recorded")
        return
    
    chosen_idx = np.linspace(0, len(snapshots)-1, min(num_snapshots, len(snapshots))).astype(int)
    cmap = plt.colormaps['viridis'].resampled(len(chosen_idx))

    if objective_labels is None:
        objective_labels = ["f1", "f2"]

    fig, ax = plt.subplots(figsize = (8,6))
    for color_i, snap_i in enumerate(chosen_idx):
        fitnesses = np.array(snapshots[snap_i])
        if fitnesses.size == 0:
            continue
        ax.scatter(fitnesses[:,0], fitnesses[:,1], color=cmap(color_i), label=f"Iteration {snap_i}", s=20, alpha=0.85)

    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_title("Pareto Front Tracking Over Time")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.show()