"""
generate_paths.py

This module provides functions to generate random paths in a graph using the NetworkX library.
It includes functions for generating random paths, shortest paths, and random routes, as well as functions for generating waypoints and complete routes.

Dependencies:
----
- networkx

Usage:
----
1. Import the required library: networkx.
2. Import this module: import generate_random_paths.
3. Use the provided functions to generate random paths in a graph.

Example:
# Generate a random path in a graph
>>> import networkx as nx
>>> from generate_random_paths import generate_random_path
>>> G = nx.DiGraph()
>>> G.add_edge(1, 2)
>>> G.add_edge(2, 3)
>>> path = generate_random_path(G, 1, 3)
>>> print(path)
[1, 2, 3]

Note:
-----
This module is part of the Graph Simulation package.
"""

import networkx as nx
import numpy as np
import pandas as pd
import random
import json


def generate_waypoints_by_basket():
    """
    Returns a list of waypoints for a random basket.

    Returns:
    --------
    - list: A list of waypoints for a random basket.
    """
    basket = get_basket()
    route = []
    while basket:
        sample = random.sample(basket, 1)[0]
        basket.remove(sample)
        if not sample:
            continue
        route.append(get_random_node(sample))
    return route


def get_shortest_route(G:  nx.Graph, waypoints: list) -> list[int]:
    """
    Returns the shortest route in the graph G that passes through the given waypoints.

    Parameters:
    -----------
    - G (networkx.Graph): The graph to find the route in.
    - waypoints (list): The list of nodes that the route must pass through.

    Returns:
    --------
    - List with the shortest route as a list of nodes.
    """
    shortest_path = []

    for i in range(len(waypoints) - 1):
        path = nx.shortest_path(
            G, source=waypoints[i], target=waypoints[i + 1])
        shortest_path.extend(path[:-1])
    shortest_path.append(waypoints[-1])

    return shortest_path


def get_random_route(G:  nx.Graph, waypoints: list) -> list[int]:
    """
    Returns a random route in the graph G that passes through the given waypoints.

    Parameters:
    -----------
    - G (networkx.Graph): The graph to find the route in.
    - waypoints (list): The list of nodes that the route must pass through.

    Returns:
    --------
    - List with the random route as a list of nodes.
    """
    random_path = []
    random_path.append(waypoints[0])

    for i in range(len(waypoints) - 1):
        current_node = waypoints[i]
        target_node = waypoints[i + 1]

        while current_node != target_node:
            neighbors = list(G.neighbors(current_node))
            next_node = random.choice(neighbors)
            random_path.append(next_node)
            current_node = next_node
    return random_path


def get_categories() -> list[str]:
    """
    Returns a list of categories from the categories_nodes_map.csv file.

    Returns:
    --------
    - list: A list of categories.
    """

    try:
        df = pd.read_csv("../data/categories_nodes_map.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/categories_nodes_map.csv")
    return list(df.columns)


def get_basket() -> list[str]:
    """
    Returns a random basket of categories from the groceries_basket_categorized.csv file.

    Returns:
    --------
    - list: A list of categories in the random basket.
    """
    try:
        df = pd.read_csv("../data/groceries_basket_categorized.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/groceries_basket_categorized.csv")

    random_row_index = np.random.choice(df.index)
    random_row = df.loc[random_row_index]
    non_nan_values = random_row.dropna().to_list()
    return non_nan_values


def get_random_node(category: str) -> int:
    """
    Returns a random node in a category.

    Parameters:
    -----------
    - category (str): The category to get the node from.

    Returns:
    --------
    - int: A random node from the specified category.
    """
    try:
        data = json.load(open("../data/categories_nodes_map.json"))
    except FileNotFoundError:
        data = json.load(open("data/categories_nodes_map.json"))
    return random.sample(data.get(category), 1)[0]


def get_waypoints(entrance_node, exit_node):
    """
    Generates waypoints for a route, including the entrance node, exit node, and a random node from the "kassa" category.

    Parameters:
    -----------
    - entrance_node (int): The node where the route starts.
    - exit_node (int): The node where the route ends.

    Returns:
    --------
    - list: A list of waypoints for the route.
    """

    waypoints = generate_waypoints_by_basket()
    waypoints.insert(entrance_node, 0)
    kassa = get_random_node("kassa")
    waypoints.append(kassa)
    waypoints.append(exit_node)

    return waypoints


def get_complete_route(G, entrance_node, exit_node):
    """
    Generates a complete route in the graph G from the entrance_node to the exit_node, passing through randomly generated waypoints.

    Parameters:
    -----------
    - G (networkx.Graph): The graph to generate the route in.
    - entrance_node (int): The node where the route starts.
    - exit_node (int): The node where the route ends.

    Returns:
    --------
    - list: The complete route as a list of nodes.
    """
    route = []
    waypoints = get_waypoints(entrance_node, exit_node)

    for node in get_shortest_route(G, waypoints):
        route.append(node)

    return_dict = {
        "route": route,
        "waypoints": waypoints
    }

    return return_dict
