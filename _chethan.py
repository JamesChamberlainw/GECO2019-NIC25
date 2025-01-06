import csv 
import numpy as np
import random
import matplotlib.pyplot as plt
from pyhv import Hypervolume  # Import pyhv for hypervolume calculation

# ... (Rest of the code remains unchanged)

def calculate_hypervolume(solutions, reference_point):
    """
    Calculates the hypervolume for a given set of solutions with respect to a reference point.
    """
    hv = Hypervolume(reference_point)
    return hv.compute(solutions)

# MAIN CODE

nodes, problem_dict = load_data(DIR + "a280-n1395.txt")

ga = GA(nodes, problem_dict, pop_size=250)

# Generation loop
for i in range(10):
    ga.generation()

    x, y = ga.gen_fitness()
    x_label = "time"
    y_label = "-profit"

    solutions = [[xi, yi] for xi, yi in zip(x, y)]

    # Get consecutive Pareto fronts
    pareto_fronts, _ = get_consecutive_pareto_fronts(solutions)

    # Hypervolume Calculation for each Pareto front
    for i, front in enumerate(pareto_fronts):
        front_np = np.array(front)
        reference_point = np.array([max(front_np[:, 0]) + 1, max(front_np[:, 1]) + 1])  # Reference point is chosen slightly worse than the worst solution in the front
        hypervolume_value = calculate_hypervolume(front_np, reference_point)
        print(f"Hypervolume of Pareto front {i+1}: {hypervolume_value}")

    # Plot all the solutions
    solutions_np = np.array(solutions)
    plt.scatter(solutions_np[:, 0], solutions_np[:, 1], color='gray', label="All solutions")

    # colormap with distinct colours
    cmap = plt.get_cmap('tab10', len(pareto_fronts)) 

    # Plot each Pareto front with a line connecting the points 
    for i, front in enumerate(pareto_fronts):
        front_np = np.array(sorted(front, key=lambda x: x[0]))
        color = cmap(i)  # Get the color for the i-th front
        plt.scatter(front_np[:, 0], front_np[:, 1], label=f'Front {i+1}', color=color)
        plt.plot(front_np[:, 0], front_np[:, 1], color=color, linestyle='-', marker='o')

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Consecutive Pareto Fronts')

    # Show legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()