"""
generate_plots.py

This script provides functions to generate and visualize plots using the NetworkX and Matplotlib libraries.
It includes functions for generating exposure map plots and graph over image plots.

Dependencies:
----
- networkx
- matplotlib
- PIL
- numpy

Usage:
----
1. Import the required libraries: networkx, matplotlib.pyplot, PIL, numpy.
2. Import this script: import generate_plots.
3. Use the provided functions to generate and visualize plots.

Example:
----
# Generate an exposure map plot
>>> from generate_plots import generate_exposure_map_plot
>>> infections_random = {1: 10, 2: 20, 3: 30}
>>> infections_path = {1: 15, 2: 25, 3: 35}
>>> generate_exposure_map_plot(infections_random, infections_path)

# Generate a graph over image plot
>>> from generate_plots import generate_graph_over_image
>>> generate_graph_over_image()

Note:
-----
This module is part of the Graph Simulation package.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import networkx as nx
import generators.generate_graph as graph_generator
import json

from PIL import Image


def generate_exposure_map_plot(infections_random: dict, infections_path: dict) -> None:
    """
    Generates an exposure map plot using the given infection data.
    The function saves the plot as a PNG image.

    Parameters:
    ----
    - infections_random (dict): A dictionary mapping nodes to infection counts for random paths.
    - infections_path (dict): A dictionary mapping nodes to infection counts for specific paths.

    """
    if infections_random['0'] == 0:
        infections_random['0'] = infections_random['65']

    if infections_path['0'] == 0:
        infections_path['0'] = infections_path['65']

    # normalize values
    max_infections = max(max(infections_random.values()),
                         max(infections_path.values()))
    min_infections = min(min(infections_random.values()),
                         min(infections_path.values()))
    infections_random = {k: (v - min_infections) / (max_infections - min_infections)
                         for k, v in infections_random.items()}
    infections_path = {k: (v - min_infections) / (max_infections - min_infections)
                       for k, v in infections_path.items()}

    # replace -inf with 0
    infections_random = {k: 0 if v == -
                         np.inf else v for k, v in infections_random.items()}
    infections_path = {k: 0 if v == -
                       np.inf else v for k, v in infections_path.items()}

    G = graph_generator.generate_graph_undirected()
    pos = graph_generator.generate_pos(G)
    H = graph_generator.generate_graph_directed()
    image_path = "images/supermarket_layout_bw_rotated.png"
    G.add_edge(103, 96)

    generate_both_plots_and_save(
        G, pos, infections_random, infections_path, image_path)

    rotate_image_and_save()


def generate_graph_over_image(color: bool = False, labels: bool = False) -> None:
    """
    Generates a plot of the graph over the supermarket image.
    The function saves the plot as a PNG image.

    Parameters:
    ---------
    - color (boolean): indicating whether to use the color or black and white image.
    - labels (boolean): indicating whether to show the node labels.
    """
    G = graph_generator.generate_graph_undirected()
    pos = graph_generator.generate_pos(G)
    if color:
        img = mpimg.imread('images/supermarket_layout_colour.png')
    else:
        img = mpimg.imread('images/supermarket_layout_bw_rotated.png')

    fig, ax = plt.subplots(figsize=(9, 22))

    x_min, x_max = min(pos.values(), key=lambda x: x[0])[
        0], max(pos.values(), key=lambda x: x[0])[0]
    y_min, y_max = min(pos.values(), key=lambda x: x[1])[
        1], max(pos.values(), key=lambda x: x[1])[1]

    ax.imshow(img, extent=[x_min-1.15, x_max+1.15,
              y_min-2.4, y_max+2.4], aspect='auto', zorder=-1)

    nx.draw(G, pos, with_labels=labels)

    plt.savefig('plots/graph_over_image.png')
    plt.show()

    original_image = Image.open('plots/graph_over_image.png')
    rotate_image = original_image.rotate(-90, expand=True)
    rotate_image.save("plots/graph_over_image.png")


def generate_single_plot(G: nx.Graph, pos: dict,
                         infections: list, image: str, ax=None, colorbar=True) -> None:
    """
    Generates a plot of the exposure time per node in the supermarket.

    Parameters:
    ---------
    - G: networkx graph
    - pos: dictionary with positions of nodes
    - infections: list of exposure times per node
    - image: path to image
    - ax: matplotlib axis
    - colorbar: boolean indicating whether to show the colorbar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 11))
    else:
        fig = plt.gcf()

    img = mpimg.imread(image)

    x_min, x_max = min(pos.values(), key=lambda x: x[0])[
        0], max(pos.values(), key=lambda x: x[0])[0]
    y_min, y_max = min(pos.values(), key=lambda x: x[1])[
        1], max(pos.values(), key=lambda x: x[1])[1]

    weights = [infections[str(n)] for n in G.nodes]

    cmap = plt.cm.YlOrRd

    norm = plt.Normalize(min(weights), max(weights))
    node_colors = cmap(norm(weights))

    alpha_values = [0 if n == 119 else 1 for n in G.nodes]
    nx.draw(G, pos, with_labels=False, node_color=node_colors,
            cmap=cmap, node_size=130, alpha=alpha_values, ax=ax)

    ax.imshow(img, extent=[x_min-1.15, x_max+1.15,
              y_min-2.4, y_max+2.4], aspect='auto', zorder=-1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if colorbar:
        cbar = plt.colorbar(sm, orientation='horizontal', pad=0.01, ax=ax)
        cbar.ax.tick_params(labelsize=8, rotation=90)
        cbar.set_label('Infections', rotation=90, labelpad=1)
    else:
        cbar = None

    return fig, ax, cbar, sm


def generate_both_plots_and_save(G: nx.Graph, pos: dict, infections_random: list,
                                 infections_path: list, image: str) -> plt.Figure:
    """
    Generates two plots of the graph G with the given node positions, one for random
    infections and one for path infections, and saves them as an image.

    Parameters:
    ----------
    - G (networkx.Graph): The graph to plot.
    - pos (dict): A dictionary mapping nodes to positions.
    - infections_random (list): A list of random infection counts for the nodes.
    - infections_path (list): A list of path infection counts for the nodes.
    - image (str): The path to the image file to plot the graph over.

    Returns:
        fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    max_infections = max(max(infections_random.values()),
                         max(infections_path.values()))

    infections_random['119'] = max_infections
    infections_path['119'] = max_infections

    G.add_node(119, pos=(1, 2))
    G.add_edge(102, 100)
    pos[119] = (1, 2)

    H = graph_generator.generate_graph_directed()
    H.add_node(119, pos=(1, 2))
    H.add_edge(102, 100)

    fig, axs = plt.subplots(1, 2, figsize=(9, 11))

    fig, ax1, _, sm = generate_single_plot(
        G, pos, infections_random, image, ax=axs[0], colorbar=False)
    fig, ax2, _, _ = generate_single_plot(
        H, pos, infections_path, image, ax=axs[1], colorbar=False)

    cax = fig.add_axes([0.15, 0.0, 0.7, 0.03])
    cbar = plt.colorbar(cax=cax, mappable=sm, orientation='horizontal')

    cbar.ax.tick_params(labelsize=8, rotation=90)
    mini = min(min(infections_random.values()), min(infections_path.values()))
    maxi = max(max(infections_random.values()), max(infections_path.values()))
    ticks = [mini, maxi]
    tick_labels = ['Low', 'High']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Relative\nexposure time', rotation=90, labelpad=1)

    plt.text(-0.05, 0.5, 'Exposure time per node in supermarket', rotation=90,
             va='center', ha='center', transform=fig.transFigure, fontsize=20)
    plt.text(0., 0.5, 'Regular layout', rotation=90, va='center',
             ha='center', transform=fig.transFigure, fontsize=12)
    plt.text(0.5, 0.5, 'Designated walking paths', rotation=90,
             va='center', ha='center', transform=fig.transFigure, fontsize=12)
    plt.tight_layout()

    fig.savefig('plots/infections_per_node.png', dpi=300, bbox_inches='tight')

    return fig


def rotate_image_and_save() -> None:
    """
    Rotates the image of the exposure time per node in the supermarket and saves it.
    """
    original_image = Image.open('plots/infections_per_node.png')
    rotate_image = original_image.rotate(-90, expand=True)
    rotate_image.save("plots/exposure_per_node.png")
    plt.show()


def plot_path_length_distribution_combined(path_lengths_random: list, path_lengths_path: list) -> None:
    """
    Plots the combined path length distribution for the given data.
    The function saves the plot as a PNG image.

    Parameters:
    ----
    - path_lengths_random (list): A list of path lengths for random paths.
        Each element in the list represents the length of a path that was randomly generated in the simulation.
    - path_lengths_path (list): A list of path lengths for specific paths.
        Each element in the list represents the length of a specific path that was used in the simulation.

    """
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        pass
    plt.figure(figsize=(8, 6))

    mean_length_random = np.mean(path_lengths_random)
    std_dev_random = np.std(path_lengths_random)
    plt.hist(path_lengths_random, bins=20, alpha=0.5,
             label='Regular layout', density=True)
    plt.axvline(mean_length_random, color='blue',
                linestyle='dashed', linewidth=2)
    plt.axvline(mean_length_random - std_dev_random,
                color='blue', linestyle='dotted', linewidth=2)
    plt.axvline(mean_length_random + std_dev_random,
                color='blue', linestyle='dotted', linewidth=2)
    plt.text(mean_length_random + 0.5, plt.ylim()
             [1] * 0.9, 'Mean: {:.2f}'.format(mean_length_random), color='blue')

    mean_length_path = np.mean(path_lengths_path)
    std_dev_path = np.std(path_lengths_path)
    plt.hist(path_lengths_path, bins=20, alpha=0.5,
             label='Designated walking paths', density=True)
    plt.axvline(mean_length_path, color='orange',
                linestyle='dashed', linewidth=2)
    plt.axvline(mean_length_path - std_dev_path,
                color='orange', linestyle='dotted', linewidth=2)
    plt.axvline(mean_length_path + std_dev_path,
                color='orange', linestyle='dotted', linewidth=2)
    plt.text(mean_length_path + 0.5, plt.ylim()
             [1] * 0.85, 'Mean: {:.2f}'.format(mean_length_path), color='orange')

    plt.xlabel('Path length (in nodes)')
    plt.ylabel('Density')
    plt.title(
        'Path length distribution for the two different layouts (with 1 standard deviation)')
    plt.legend()
    plt.savefig('plots/path_length_distribution_combined.png')
    plt.show()


def plot_infection_distribution(data) -> None:
    """
    Plots the distribution of infections for the two different layouts.

    Parameters:
    ----
    - data (dict): A dictionary containing infection data for the two layouts. 
                     The dictionary should have keys 'directed' and 'undirected', 
                     each containing a sub-dictionary with the key 'get_total_infected' 
                     which contains another sub-dictionary with the key 'total infected' 
                     mapping to a list of total infections for each simulation run.
    """

    data_path = data['directed']['get_total_infected']['total infected']
    data_random = data['undirected']['get_total_infected']['total infected']

    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        pass
    plt.figure(figsize=(8, 6))

    mean_infections_random = np.mean(data_random)
    std_infections_random = np.std(data_random)
    plt.hist(data_random, bins=15, alpha=0.5, label='Regular layout')
    plt.axvline(mean_infections_random, color='blue', linestyle='dashed',
                linewidth=2)
    plt.axvline(mean_infections_random - std_infections_random, color='blue',
                linestyle='dotted', linewidth=1)
    plt.axvline(mean_infections_random + std_infections_random, color='blue',
                linestyle='dotted', linewidth=1)
    plt.text(mean_infections_random + 0.5, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_infections_random),
             color='blue')

    mean_infections_path = np.mean(data_path)
    std_infections_path = np.std(data_path)
    plt.hist(data_path, bins=14, alpha=0.5, label='Designated walking paths')
    plt.axvline(mean_infections_path, color='orange', linestyle='dashed',
                linewidth=2)
    plt.axvline(mean_infections_path - std_infections_path, color='orange',
                linestyle='dotted', linewidth=1)
    plt.axvline(mean_infections_path + std_infections_path, color='orange',
                linestyle='dotted', linewidth=1)
    plt.text(mean_infections_path + 0.5, plt.ylim()[1] * 0.8, 'Mean: {:.2f}'.format(mean_infections_path),
             color='orange')

    plt.title('Distribution of Infections (with 1 standard deviation)')
    plt.xlabel('Number of Infections')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()
