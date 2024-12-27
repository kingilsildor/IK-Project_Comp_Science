"""
sensitivity_analysis.py

This module provides sensitivity analysis, comparing the change in infections in the two simulations.

Dependencies:
----
- seaborn
- numpy
- pandas
- matplotlib
- SALib

"""
import SALib
from classes.simulation import Simulation
from matplotlib.colors import ListedColormap
from SALib.sample.sobol import sample
from SALib.analyze import sobol

import generators.generate_graph as graph_generator
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def run_analysis(params: dict[float]) -> float:
    """
    Run a simulation analysis comparing the spread of infections in a supermarket
    with and without a designated route for agents.

    Parameters:
    -----
    - params (dict): A dictionary containing simulation parameters.
        - 'n_agent' (float): Number of agents in the simulation.
        - 'timesteps' (float): Number of simulation timesteps.
        - 'infecting_init' (float): Initial number of infected agents.
        - 'infecting_prob' (float): Probability of infection.

    Returns:
    ----
    - float: The difference in the percentage change of infections between the two scenarios.
        - Positive value indicates the scenario with the designated route is better.
        - Negative value indicates the scenario without the designated route is better.
        - Zero indicates no significant difference between the scenarios.

    The function conducts two simulations, one without a designated route (G_supermarket)
    and one with a designated route (H_supermarket). It then calculates and prints the percentage
    change in infections for each scenario and the difference between them. Finally, it determines
    which scenario is better based on the percentage change in infections and returns the difference.
    """
    iterations = 1
    params['n_agent'] = int(params['n_agent'])
    params['timesteps'] = int(params['timesteps'])
    print(params)

    # generate the supermarket
    # without designated route
    G = graph_generator.generate_graph_undirected()

    # with designated route
    H = graph_generator.generate_graph_directed()
    posG = graph_generator.generate_pos(G)
    posH = graph_generator.generate_pos(H)

    # setup simulation
    G_supermarket = Simulation(G, posG, timesteps=params['timesteps'], n_agent=params['n_agent'], visualize=False,
                               infecting_init=params['infecting_init'], infecting_adjacency=1, infecting_prob=params['infecting_prob'])
    G_supermarket.simulate(iterations)

    H_supermarket = Simulation(H, posH, timesteps=params['timesteps'], n_agent=params['n_agent'], visualize=False,
                               infecting_init=params['infecting_init'], infecting_adjacency=1, infecting_prob=params['infecting_prob'])
    H_supermarket.simulate(iterations)

    print("G_supermarket.get_total_infected()",
          G_supermarket.get_total_infected())
    print("H_supermarket.get_total_infected()",
          H_supermarket.get_total_infected())

    # measure % change in infections
    change_G = G_supermarket.get_total_infected()['total infected'][0]
    total_G = G_supermarket.get_total_infected()['total cases'][0]

    change_H = H_supermarket.get_total_infected()['total infected'][0]
    total_H = H_supermarket.get_total_infected()['total cases'][0]

    HR_G = (total_G-change_G) and change_G / (total_G-change_G) or 0
    HR_G *= 100

    HR_H = (total_H-change_H) and change_H / (total_H-change_H) or 0
    HR_H *= 100

    # compare % change in infections
    print(f"Without route: {HR_G}% change.")
    print(f"With route: {HR_H}% change")
    print(f"difference: {HR_G - HR_H}")

    # difference in change in cases between the two paths
    diff = HR_G - HR_H

    if diff > 0:
        print("With route is better")
    elif diff < 0:
        print("Without route is better")

    if not diff:
        return 0
    return diff


def run() -> None:
    """
    Perform a sensitivity analysis on a simulation model.

    This function generates samples for specified parameter ranges, runs the simulation model
    for each set of parameters, and conducts a sensitivity analysis using Sobol indices. It
    visualizes the results through bar plots, histograms, scatter plots, and violin plots,
    and saves the sensitivity analysis data to CSV files.

    The sensitivity analysis considers the direct impact of individual parameters on the model output
    (First-Order Sensitivity) and the total impact, including interactions with other parameters
    (Total-Order Sensitivity).

    The function also generates interaction plots, scatter plots, and parameter distribution plots.

    The final results and plots are saved in the 'images' directory and CSV files in the 'analysis/data' directory.

    Note: Ensure the necessary libraries (numpy, scipy, seaborn, matplotlib, and SALib) are installed before running.

    """
    # define the param ranges for the problem
    names = ['timesteps', 'n_agent', 'infecting_init', 'infecting_prob']
    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': [[100, 1500], [100, 1000], [0.00001, 0.05], [0.0000141, 0.01]],
        'dists': ['unif', 'unif', 'unif', 'unif']
    }

    # generate samples
    param_values = sample(problem, 4)
    print("param_values", param_values)
    counter = 0

    # Run model
    result = np.array([])
    for iteration in param_values:
        counter += 1
        param_dict = {names[pos]: iteration[pos] for pos in range(len(names))}
        result = np.append(result, np.array(
            [[run_analysis({names[pos]: iteration[pos] for pos in range(len(names))})]]))
        print(f"""
        ----------------------------------------
        iteration {counter} / {len(param_values)+1} \n
        result : {result} \n
        with parameter values : {param_dict}
        ----------------------------------------
        """)

    Si = sobol.analyze(
        problem, result, calc_second_order=True, print_to_console=True)

    total_Si, first_Si, second_Si = Si.to_df()

    plt.close('all')

    param_samples = param_values
    model_outputs = result

    # Sensitivity analysis
    Si = sobol.analyze(problem, model_outputs, calc_second_order=True)

    # Extract sensitivity indices
    param_names = problem['names']
    first_order_indices = Si['S1']
    total_order_indices = Si['ST']

    # Create a Seaborn figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # Bar plot for first-order sensitivity indices
    # Measures the direct impact of an individual parameter on the output, ignoring interactions with other parameters
    sns.barplot(x=param_names, y=first_order_indices,
                ax=axes[0, 0], palette=sns.color_palette("flare"))
    axes[0, 0].set_title('First-Order Sensitivity')

    # Bar plot for total-order sensitivity indices
    # Measures the total impact of a parameter on the output, considering both its direct effects and interactions with other parameters.
    sns.barplot(x=param_names, y=total_order_indices,
                ax=axes[0, 1], palette=sns.color_palette("flare"))
    axes[0, 1].set_title('Total-Order Sensitivity')

    # Histograms
    # plot the frequency of the model outputs
    sns.histplot(model_outputs, bins='auto', kde=True,
                 ax=axes[0, 2], palette=sns.color_palette("flare"))
    axes[0, 2].set_title('Model Output Frequency')
    axes[0, 2].set_xlabel('Model Output')
    axes[0, 2].set_ylabel('Frequency')

    # Scatter plots
    # plot different parameters vs model output
    axes[1, 1].scatter(param_samples[:, 0], model_outputs, label=param_names[0],
                       cmap=ListedColormap(sns.color_palette("flare").as_hex()))
    axes[1, 1].scatter(param_samples[:, 1], model_outputs, label=param_names[1],
                       cmap=ListedColormap(sns.color_palette("flare").as_hex()))
    axes[1, 1].set_title('Interaction Plot')
    axes[1, 1].set_xlabel('Parameter Value')
    axes[1, 1].set_ylabel('Model Output')
    axes[1, 1].legend()

    axes[1, 0].scatter(param_samples[:, 2], model_outputs, label=param_names[2],
                       cmap=ListedColormap(sns.color_palette("flare").as_hex()))
    axes[1, 0].scatter(param_samples[:, 3], model_outputs, label=param_names[3],
                       cmap=ListedColormap(sns.color_palette("flare").as_hex()))
    axes[1, 0].set_title('Interaction Plot')
    axes[1, 0].set_xlabel('Parameter Value')
    axes[1, 0].set_ylabel('Model Output')
    axes[1, 0].legend()

    # plot distributions
    data = pd.DataFrame(param_samples, columns=names)
    x = 1
    y = 2
    for i in range(len(names)):
        sns.violinplot(x=data[names[i]], ax=axes[x, y], native_scale=True, fill=False, inner_kws=dict(
            box_width=15, whis_width=2, color=".8"), palette=sns.color_palette("flare").as_hex())
        axes[x, y].set_title(f'{names[i]} Distribution')
        y += 1
        if y > 2:
            x += 1
            y = 0
        if x > 2:
            break
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/sensitivity_analysis.png')
    plt.show()

    total_Si.to_csv("analysis/data/total_Si.csv")
    first_Si.to_csv("analysis/data/first_Si.csv")
    second_Si.to_csv("analysis/data/second_Si.csv")
