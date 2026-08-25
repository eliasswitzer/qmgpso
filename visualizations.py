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
        ax.scatter(fitnesses[:,0], fitnesses[:,1], c='tab:blue', s=10)
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    elif num_objectives == 3:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fitnesses[:,0], fitnesses[:,1], fitnesses[:,2], c='tab:blue', s=10)
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

def plot_pareto_front_history(history, true_history=None, objective_labels=None, num_snapshots=5, tau_T=None):
    """Overlays several archive snapshots to show how the Pareto front changed over time from the changing environment"""
    snapshots = history['archive_snapshots']
    if not snapshots:
        print("No archive snapshots recorded")
        return
    
    if tau_T is not None:
        chosen_idx = [i for i in range(tau_T-1, len(snapshots), tau_T)]
        chosen_idx = chosen_idx[:num_snapshots] if len(chosen_idx) > num_snapshots else chosen_idx
    else:
        chosen_idx = np.linspace(0, len(snapshots)-1, min(num_snapshots, len(snapshots))).astype(int)

    num_objectives = None
    for i in chosen_idx:
        arr = np.array(snapshots[i])
        if arr.size > 0:
            num_objectives = arr.shape[1]
            break
    if num_objectives is None and true_history is not None:
        for i in chosen_idx:
            if i < len(true_history):
                arr = np.array(true_history[i])
                if arr.size > 0:
                    num_objectives = arr.shape[1]
                    break

    if num_objectives is None:
        print("All selected snapshots are empty, nothing to plot")
        return

    if num_objectives not in (2, 3):
        print("Too many objectives for a plot")
        return

    if objective_labels is None:
        objective_labels = [f"f{i+1}" for i in range(num_objectives)]

    fig = plt.figure(figsize = (8,8))
    if num_objectives == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    for idx, i in enumerate(chosen_idx):
        fitnesses = np.array(snapshots[i])
        if fitnesses.size == 0:
            continue

        if num_objectives == 3:
            ax.scatter(fitnesses[:,0], fitnesses[:,1], fitnesses[:,2], color='tab:blue', label=f"Iteration {i+1}", s=10, alpha=0.85)
        else:
            ax.scatter(fitnesses[:,0], fitnesses[:,1], color='tab:blue', label=f"Iteration {i+1}", s=10, alpha=0.85)

        if true_history is not None and i < len(true_history):
            true_pof = np.array(true_history[i])
            if true_pof.size == 0:
                continue

            if num_objectives == 3:
                ax.scatter(true_pof[:,0], true_pof[:,1], true_pof[:,2], color='tab:red', marker='x', s=15, alpha=0.4, zorder=1)
            else:
                order = np.argsort(true_pof[:,0])
                ax.plot(true_pof[order,0], true_pof[order,1], color='tab:red', linewidth=1.5, alpha=0.5, zorder=1)

    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if num_objectives == 3:
        ax.set_zlabel(objective_labels[2])
        ax.set_zlim(0.0, 1.0)
    ax.set_title("Pareto Front Tracking Over Time")
    ax.grid(True)
    plt.tight_layout()
    plt.show()