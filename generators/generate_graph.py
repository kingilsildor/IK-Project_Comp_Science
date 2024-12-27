"""
graph_generator.py

This script provides functions to generate and visualize graphs using the NetworkX and Matplotlib libraries.
It includes functions for generating undirected and directed graphs, obtaining node positions, and creating visualizations.

Dependencies:
----
- networkx
- matplotlib

Usage:
----
1. Import the required libraries: networkx, matplotlib.pyplot.
2. Import this script: import graph_generator.
3. Use the provided functions to generate and visualize graphs.

Example:
----
# Generate an undirected graph
undirected_graph = graph_generator.generate_graph_undirected()

# Visualize the undirected graph
graph_generator.generate_picture(undirected_graph)

# Generate a directed graph
directed_graph = graph_generator.generate_graph_directed()

# Visualize the directed graph
graph_generator.generate_picture(directed_graph)

Note:
-----
This module is part of the Graph Simulation package.
"""

import networkx as nx
import matplotlib.pyplot as plt

def generate_graph_undirected() -> nx.Graph:
    """
    Generate an undirected graph.

    Returns
    -------
    nx.Graph
        An undirected graph with nodes and edges.

    Example
    -------
    >>> G = generate_graph_undirected()
    """
    G = nx.Graph()
    add_nodes(G)
    add_edges_undirected(G)
    return G

def generate_graph_directed() -> nx.DiGraph:
    """
    Generate a directed graph.

    Returns
    -------
    nx.DiGraph
        A directed graph with nodes and edges.

    Example
    -------
    >>> G = generate_graph_directed()
    """
    G = nx.DiGraph()
    add_nodes(G)
    add_edges_directed(G)
    return G

def generate_pos(G) -> dict:
    """
    Generate node positions for a graph.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The input graph.

    Returns
    -------
    dict
        A dictionary mapping node identifiers to positions.
    """
    return nx.get_node_attributes(G, 'pos')

def generate_picture(G) -> None:
    """
    Generate a visualization of the graph.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The input graph.
    """
    plt.figure(figsize=(9,22))
    pos = generate_pos(G)
    nx.draw(G, pos, with_labels=True)
    if isinstance(G, nx.classes.digraph.DiGraph):
        plt.savefig("../plots/graph_directed.png", format="PNG")
    else: 
        plt.savefig("../plots/graph.png", format="PNG")
    return

def add_edges_directed(G) -> None:
    """
    Add directed edges to the graph.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph.
    """
    edges = [
        (0,65),(65,30),(30,49),(49,48),(48,47),(47,45),(45,44),(44,43),(43,42),(42,41),(41,39),
        (39,38),(38,37),(37,36),(36,35),(35,34),(34,32),(32,31),(31,29),(29,28),(28,27),(27,26),
        (26,25),(25,19),(19,18),(18,17),(17,16),(16,15),(15,14),(14,13),(13,12),(12,11),(11,10),
        (10,9),(9,8),(8,7),(7,6),(6,5),(5,4),(4,3),(3,2),(2,1),(1,65),(45,46),(46,3),(8,40),(40,39),
        (34,33),(33,13),(19,20),(20,19),(21,20),(20,21),(21,22),(22,21),(22,23),(23,22),(23,24),
        (24,23),(26,96),(96,98),(98,99),(99,100),(100,101),(101,102),(102,103),(34,103),(103,76),
        (76,75),(75,81),(81,82),(82,83),(83,84),(84,85),(85,86),(86,87),(87,88),(88,89),(89,90),
        (90,91),(90,93),(93,94),(94,95),(95,96),(84,91),(91,94),(75,77),(77,78),(78,80),(80,67),
        (67,69),(69,70),(70,71),(71,72),(72,79),(79,36),(75,74),(74,71),(71,73),(73,62),(66,67),
        (62,63),(63,64),(64,66),(39,61),(61,60),(60,59),(60,62),(62,59),(59,58),(58,57),(57,56),
        (56,55),(55,42),(56,54),(54,55),(54,57),(48,50),(50,51),(54,53),(53,52),(52,51),(104,53),
        (104,54),(51,105),(105,104),(105,110),(110,111),(111,110),(111,112),(112,111),(110,113),
        (113,114),(110,106),(106,105),(107,106),(109,107),(109,108),(108,107),(115,109),(114,115),
        (115,116),(114,116),(116,117),(117,118)
    ]
    G.add_edges_from(edges)
    return

def add_edges_undirected(G) -> None:
    """
    Add undirected edges to the graph.

    Parameters
    ----------
    G : nx.Graph
        The undirected graph.
    """
    edges = [
        (0, 65), (65, 1), (1, 2), (2, 3), (3, 4), (3, 46), (4, 5), (5, 6), (6, 7), (7, 8), (8, 40), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 33), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21),
        (21, 22), (22, 23), (23, 24), (19, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 31), (31, 32), (32, 34), (33, 34),
        (34, 35), (35, 36), (36, 37), (38, 39), (39, 61), (39, 40), (39, 41), (41, 42), (42, 55), (42, 43), (44, 45), (43, 44),
        (45, 46), (45, 47), (47, 48), (48, 49), (48, 50), (50, 51), (51, 105), (51, 52), (52, 53), (53, 54), (54, 55), (54, 56),
        (54, 57), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (59, 62), (60, 61), (60, 62), (62, 63), (63, 64), (64, 66),
        (66, 67), (67, 69), (67, 80), (69, 70), (70, 71), (71, 72), (71, 73), (72, 79), (62, 73), (71, 74), (74, 75), (75, 76),
        (75, 77), (77, 78), (78, 80), (75, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89),
        (89, 90), (90, 93), (90, 91), (91, 94), (91, 84), (93, 94), (94, 95), (95, 96), (96, 98), (96, 26), (98, 99), (99, 100),
        (100, 101), (101, 102), (102, 103), (103, 76), (76, 34), (53, 104), (104, 105), (105, 106), (106, 107), (107, 108),
        (107, 109), (108, 109), (109, 115), (105, 110), (106, 110), (110, 111), (110, 112), (110, 113), (113, 114), (114, 115),
        (114, 116), (115, 116), (116, 117), (117, 118), (37, 38), (79, 36), (65, 30), (30, 49), (0, 65),(104,54)
    ]
    G.add_edges_from(edges)
    return

def add_nodes(G) -> None:
    """
    Add nodes to the graph with positions.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The input graph.
    """
    G.add_node(0, pos=(1,2))
    G.add_node(65, pos=(1,3))
    G.add_node(1, pos=(1,4))
    G.add_node(2, pos=(1,5))
    G.add_node(3, pos=(1,6))
    G.add_node(4, pos=(1, 7))
    G.add_node(5, pos=(1, 8))
    G.add_node(6, pos=(1,9))
    G.add_node(7, pos=(1, 10))
    G.add_node(8, pos=(1, 11))
    G.add_node(9, pos=(1, 12))
    G.add_node(10, pos=(1, 13))
    G.add_node(11, pos=(1, 14))
    G.add_node(12, pos=(1, 15))
    G.add_node(13, pos=(1, 16))
    G.add_node(14, pos=(1, 17))
    G.add_node(15, pos=(1, 18))
    G.add_node(16, pos=(1, 19))
    G.add_node(17, pos=(1, 20))
    G.add_node(18, pos=(1, 21))
    G.add_node(19, pos=(1, 22))
    G.add_node(20, pos=(0, 22))
    G.add_node(21, pos=(0, 23))
    G.add_node(22, pos=(1, 23))
    G.add_node(23, pos=(2, 23))
    G.add_node(24, pos=(3, 23))
    G.add_node(25, pos=(2, 22))
    G.add_node(26, pos=(3, 22))
    G.add_node(27, pos=(3, 21))
    G.add_node(28, pos=(3, 20))
    G.add_node(29, pos=(3, 19))
    G.add_node(30, pos=(2, 3))
    G.add_node(31, pos=(3, 18))
    G.add_node(32, pos=(3, 17))
    G.add_node(33, pos=(2, 16))
    G.add_node(34, pos=(3, 16))
    G.add_node(35, pos=(3, 15))
    G.add_node(36, pos=(3, 14))
    G.add_node(37, pos=(3, 13))
    G.add_node(38, pos=(3, 12))
    G.add_node(39, pos=(3, 11))
    G.add_node(40, pos=(2, 11))
    G.add_node(41, pos=(3, 10))
    G.add_node(42, pos=(3, 9))
    G.add_node(43, pos=(3, 8))
    G.add_node(44, pos=(3, 7))
    G.add_node(45, pos=(3, 6))
    G.add_node(46, pos=(2, 6))
    G.add_node(47, pos=(3, 5))
    G.add_node(48, pos=(3, 4))
    G.add_node(49, pos=(3, 3))
    G.add_node(50, pos=(4, 4))
    G.add_node(51, pos=(5, 4))
    G.add_node(52, pos=(5, 5))
    G.add_node(53, pos=(5, 6))
    G.add_node(54, pos=(5, 7))
    G.add_node(55, pos=(4, 9))
    G.add_node(56, pos=(5, 9))
    G.add_node(57, pos=(6, 9))
    G.add_node(58, pos=(6, 10))
    G.add_node(59, pos=(6, 11))
    G.add_node(60, pos=(5, 11))
    G.add_node(61, pos=(4, 11))
    G.add_node(62, pos=(6, 12))
    G.add_node(63, pos=(7, 12))
    G.add_node(64, pos=(8, 12))
    G.add_node(66, pos=(9, 12))
    G.add_node(67, pos=(9, 14))
    G.add_node(69, pos=(8, 14))
    G.add_node(70, pos=(7, 14))
    G.add_node(71, pos=(6, 14))
    G.add_node(72, pos=(5, 14))
    G.add_node(73, pos=(6, 13))
    G.add_node(74, pos=(6, 15))
    G.add_node(75, pos=(6, 16))
    G.add_node(76, pos=(5, 16))
    G.add_node(77, pos=(7, 16))
    G.add_node(78, pos=(8, 16))
    G.add_node(79, pos=(4, 14))
    G.add_node(80, pos=(9, 16))
    G.add_node(81, pos=(6, 17))
    G.add_node(82, pos=(6, 18))
    G.add_node(83, pos=(6, 19))
    G.add_node(84, pos=(6, 20))
    G.add_node(85, pos=(7, 20))
    G.add_node(86, pos=(8, 20))
    G.add_node(87, pos=(9, 20))
    G.add_node(88, pos=(9, 21))
    G.add_node(89, pos=(8, 21))
    G.add_node(90, pos=(7, 21))
    G.add_node(91, pos=(6, 21))
    G.add_node(93, pos=(7, 22))
    G.add_node(94, pos=(6, 22))
    G.add_node(95, pos=(5, 22))
    G.add_node(96, pos=(4, 22))
    G.add_node(98, pos=(4, 21))
    G.add_node(99, pos=(4, 20))
    G.add_node(100, pos=(4, 19))
    G.add_node(101, pos=(4, 18))
    G.add_node(102, pos=(4, 17))
    G.add_node(103, pos=(4, 16))
    G.add_node(104, pos=(6, 5))
    G.add_node(105, pos=(6, 4))
    G.add_node(106, pos=(7, 4))
    G.add_node(107, pos=(8, 4))
    G.add_node(108, pos=(9, 5))
    G.add_node(109, pos=(9, 4))
    G.add_node(110, pos=(6, 3))
    G.add_node(111, pos=(6, 2))
    G.add_node(112, pos=(6, 1))
    G.add_node(113, pos=(7, 2))
    G.add_node(114, pos=(8, 2))
    G.add_node(115, pos=(9, 3))
    G.add_node(116, pos=(9, 2))
    G.add_node(117, pos=(9, 1))
    G.add_node(118, pos=(8, 1))
    return