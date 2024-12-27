"""
This is the main script for simulating the impact of 
dedicated walking paths on COVID-19 transmission in a supermarket.

Given the global challenges posed by infectious diseases,
it's crucial to understand the impact of preventive measures.
This simulation project draws inspiration
from research such as the paper by Ying & O'Clery (2021),
which models COVID-19 transmission in supermarkets using an agent-based approach.

The primary goal is to explore how much dedicated walking paths can
reduce the probability of COVID-19 infection in a supermarket.
These paths are designed to minimize close physical interactions,
thereby mitigating the spread of infectious diseases.

Simulation Parameters:
- Framework: Supermarket Simulation with specified dimensions
- Graph Generation: Random graph representing the store layout
- Path Generation: Random path for a customer within the generated graph
- Agent: One customer with an arrival time and a generated path
- Simulation: Uses the Framework, Graph, and Agent to simulate disease transmission dynamics
- Comparison: Focuses on contrasting scenarios with and without dedicated walking paths

Please note that this simulation project builds on insights from the paper by Ying & O'Clery (2021)
and uses a network-based approach to study the effectiveness of dedicated walking paths.
While our approach is different,
we have gained valuable insights from the referenced paper to inform our model.

Author: Mees, Thomas & Tycho
Date: 29-01-2024
"""
import json
import unittest
import networkx as nx
import classes.test_protocol as test_protocol
import generators.generate_graph as graph_generator
import generators.generate_plots as plot_generator
import generators.generate_paths as path_generator

from classes.framework import Framework
from classes.simulation import Simulation
from classes.validator import FrameworkValidator, SimulationValidator


def get_json():
    """Example how to read the JSON file"""
    with open('data/simulation_data.json') as f:
        data = json.load(f)
        print(data['directed']['get_average_infected']['average cases'])


def run_plots(data: dict, directed: nx.Graph, undirected: nx.Graph):
    """
    Generates several plots based on the provided simulation data.

    Parameters:
    ----------
    - data (dict): A dictionary containing simulation data. The dictionary should have keys 'directed' and 'undirected', 
                     each containing a sub-dictionary with various simulation results.

    Function Calls:
    ---------------
    - generate_exposure_map_plot: Generates a plot showing the exposure of each node in the graph.
        It uses the average exposure data from the 'directed' and 'undirected' simulations.
    - generate_graph_over_image: Generates a plot of the graph over an image of the physical layout.
    - get_complete_route: Gets the complete route from the source node to the destination node in the graph. 
        It is used to generate path lengths for both 'directed' and 'undirected' simulations.
    - plot_path_length_distribution_combined: Generates a plot showing the distribution of path lengths for both random
        and specific paths. It uses the path lengths generated from the 'get_complete_route' function.
    - plot_infection_distribution: Generates a plot showing the distribution of infections for the two different layouts.
        It uses the infection data from the 'directed' and 'undirected' simulations.
    """
    exposure_path = data['directed']['get_node_exposure']['average']
    exposure_random = data['undirected']['get_node_exposure']['average']
    plot_generator.generate_exposure_map_plot(exposure_random, exposure_path)
    plot_generator.generate_graph_over_image()

    path_lengths_random = [len(path_generator.get_complete_route(
        directed, 0, 118)['route']) for _ in range(1000)]
    path_lengths_path = [len(path_generator.get_complete_route(
        undirected, 0, 118)['route']) for _ in range(1000)]
    plot_generator.plot_path_length_distribution_combined(
        path_lengths_random, path_lengths_path)

    plot_generator.plot_infection_distribution(data)


def run_sensitivity_analysis():
    """ Run sensitivity analysis."""
    import sensitivity_analysis
    sensitivity_analysis.run()


def run_unit_test() -> None:
    """Function for running the unit test."""
    print("\nRun unit test:")
    suite = unittest.TestLoader().loadTestsFromModule(test_protocol)
    unittest.TextTestRunner(verbosity=2).run(suite)


def run_simulation(graph: nx.Graph, config: dict, visualization: bool = False) -> None:
    """Simulate multiple graphs with the same parameters."""
    framework = None
    if visualization:
        framework = Framework(title="Social Distance", height=500, width=500)
        FrameworkValidator(framework).validate()

    pos = graph_generator.generate_pos(graph)
    supermarked = Simulation(graph, pos, frame=framework, visualize=visualization,
                             timesteps=config['timesteps'],
                             n_agent=config['n_agents'],
                             infecting_adjacency=config['infecting_adjacency'],
                             infecting_init=config['infected_init'],
                             infecting_prob=config['infecting_prob'],
                             max_node_weight=config['max_node_weight']
                             )
    SimulationValidator(supermarked).validate()
    supermarked.simulate(config["iterations"])
    supermarked.get_all_data()
    supermarked.write_json_data()


def main():
    config = {
        # Params changed to run a quicker simulation", see report for more info.
        "iterations": 1,
        "timesteps": 10,
        "n_agents": 5,
        # Params that doesn't change
        "infected_init": 1.87 * 10**-2,
        "infecting_prob": 1.41 * 10**-5,
        "infecting_adjacency": 1,
        "max_node_weight": 10
    }

    DIRECTED = graph_generator.generate_graph_directed()
    UNDIRECTED = graph_generator.generate_graph_undirected()
    for graph in [DIRECTED, UNDIRECTED]:
        run_simulation(graph, config)

    # For if the user wants to use the visualization
    # run_simulation(H, config, visualization=True)

    with open('data/__data_used_for_plots.json') as data_file:
        data = json.load(data_file)
    run_plots(data, DIRECTED, UNDIRECTED)


if __name__ == "__main__":
    main()
    run_unit_test()
    # run_sensitivity_analysis()
