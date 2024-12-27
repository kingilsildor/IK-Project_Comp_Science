"""
simulation.py

This script contains the Simulation class which represents a simulation of
agents moving through a network graph.
The class includes methods for running the simulation, collecting data,
and visualizing the simulation.

Dependencies:
----
- random
- json
- networkx
- numpy
- matplotlib

Usage:
----
1. Import the required libraries: random, json, networkx, numpy, matplotlib.
2. Import the Simulation class and other necessary classes from this file.
3. Create an instance of the Simulation class and use its methods to run a simulation.

Example:
----
# Create a networkx graph
G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(2, 3)

# Create a Simulation instance
sim = Simulation(G, {1: (0, 0), 2: (1, 1), 3: (2, 2)}, 10)

# Run the simulation
sim.run()
"""

import random
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import generators.generate_paths as path_generator

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tqdm import tqdm

from classes.agent import Agent
from classes.framework import Framework, Button
from classes.validator import AgentValidator


class Simulation:
    """
    Class representing a simulation of agents moving through a network graph.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph representing the environment.
    pos : dict
        Dictionary of node positions for visualization.
    timesteps : int
        Duration of the simulation in time steps.
    agents : list[Agent], optional (default=[])
        List of Agent objects moving through the network.
    interval : int, optional (default=300)
        Time interval for animation update in milliseconds.
    write_data : bool, optional (default=True)
        If True, write simulation data to a JSON file.
    visualize : bool, optional (default=False)
        If True, visualize the simulation using matplotlib.
    frame : Framework, optional (default=None)
        Framework instance for visualization.
    iterations : int, optional (default=1)
        Number of iterations to run the simulation.
    n_agent : int, optional (default=50)
        Number of agents in the simulation.
    infecting_prob : float, optional (default=1.41 * 10**-9)
        Probability of infection between agents.
        Must be between 0 and 1.
    infecting_init : float, optional (default=0.0187)
        Initial infection rate among agents.
        Must be between 0 and 1.
    infecting_adjacency : int, optional (default=1)
        Radius for considering adjacent nodes for infection.
    max_node_weight : int, optional (default=5)
        Maximum weight assigned to a node.

    Attributes
    ----------
    frame : Framework
        Framework instance for visualization.
    iterations : int
        Number of iterations to run the simulation.
    timesteps : int
        Duration of the simulation in time steps.
    graph : nx.Graph
        NetworkX graph representing the environment.
    graph_type : str
        Type of the graph, either "directed" or "undirected".
    pos : dict
        Dictionary of node positions for visualization.
    visualize : bool
        If True, visualize the simulation using matplotlib.
    interval : int
        Time interval for animation update in milliseconds.
    axes : plt.Axes
        Matplotlib axes for visualization.
    canvas : FigureCanvasTkAgg
        Matplotlib canvas for visualization.
    pause : bool
        Indicates whether the animation is paused.
    agents_list : list[Agent]
        List of Agent objects moving through the network.
    n_agents : int
        Number of agents in the simulation.
    infecting_prob : float
        Probability of infection between agents.
    infecting_init : float
        Initial infection rate among agents.
    infecting_adjacency : int
        Radius for considering adjacent nodes for infection.
    max_node_weight : int
        Maximum weight assigned to a node.
    write_data : bool
        If True, write simulation data to a JSON file.
    __node_weight_sum : dict
        Sum of weights for each node.
    __node_infected_sum : dict
        Sum of infected counts for each node.
    __node_exposure_sum : dict
        Sum of exposure counts for each node.
    __agent_exposure_time : dict
        Exposure time for each agent at each timestep.

    Methods
    -------
    simulate(amount: int) -> None
        Run the simulation for the specified number of iterations.
    get_total_infected() -> dict
        Get a tuple containing lists of total infected and total cases for each iteration.
    get_average_infected() -> dict
        Get a tuple containing the average number of infected and average cases over all iterations.
    get_node_weight() -> dict
        Get a dictionary containing average and total node weights.
    get_node_infected() -> dict
        Get a dictionary containing average and total infected node counts.
    get_node_exposure() -> dict
        Get a dictionary containing average and total node exposure counts.
    get_agent_exposure() -> dict
        Get a dictionary containing average and total agent exposure times.
    get_all_data() -> list[dict] | bool
        Get all the data at once in a list.
    write_json_data() -> bool
        Write the data to the JSON file.
    """
    __ROUNDING_DECIMAL = 2
    __ROUNDING_PERSON = 0
    __ROUNDING_AGENT_ACTIONS = 4

    __JSON_FILE = "data/simulation_data.json"
    __JSON_INDENT = 4

    def __init__(self, graph: nx.Graph, pos: dict, timesteps: int, agents: list[Agent] = [],
                 interval: int = 300, write_data: bool = True,
                 visualize: bool = False, frame: Framework = None, iterations: int = 1,
                 n_agent: int = 50, infecting_prob: float = 1.41 * 10**-9,
                 infecting_init: float = 0.0187, infecting_adjacency: int = 1, max_node_weight: int = 5) -> None:
        """
        Initialize the Simulation instance.

        Parameters:
        ---------
        - graph (nx.Graph): NetworkX graph representing the environment.
        - pos (dict): Dictionary of node positions for visualization.
        - frame (Framework): Framework instance for visualization. Default: None.
        - timesteps (int): Duration of the simulation in time steps.
        - agents (list[Agent]): List of Agent objects moving through the network. Default: empty list.
        - interval (int): Time interval for animation update in milliseconds. Default: 300.
        - visualize (bool): If True, visualize the simulation using matplotlib. Default: False.
        - iterations (int): Number of iterations to run the simulation. Default: 1.
        - n_agent (int): Number of agents in the simulation. Default: 50.
        - infecting_prob (float): Probability of infection between agents.
            Must be between 0 and 1. Default: 1.41 * 10**-9.
        - infecting_init (float): Initial infection rate among agents.
            Must be between 0 and 1. Default: 0.0187.
        - infecting_adjacency (int): Radius for considering adjacent nodes for infection. Default: 1.
        - max_node_weight (int): Maximum weight assigned to a node. Default: 5.
        """

        # variables for running the simulation
        self.frame: Framework = frame
        if visualize and self.frame is None:
            raise ValueError(
                "Error: If 'visualize' is True, 'framework' cannot be None.")
        self.iterations: int = iterations
        self.timesteps: int = timesteps
        self.graph: nx.Graph = graph
        self.graph_type = "directed" if nx.is_directed(
            self.graph) else "undirected"
        self.pos: dict = pos

        # variables for running the visualization
        self.visualize: bool = visualize
        self.interval: int = interval
        if self.visualize:
            self.axes: plt.Axes = frame.get_axis()
            self.canvas: FigureCanvasTkAgg = frame.get_canvas()
            self.pause: int = False
            self.__get_buttons()

        # parameters for the simulation
        if agents is not None:
            self.agents_list: list[Agent] = agents
        else:
            self.agents_list: list[Agent] = []
        self.n_agents: int = n_agent
        self.infecting_prob: float = infecting_prob
        self.infecting_init: float = infecting_init
        self.infecting_adjacency: int = infecting_adjacency
        self.max_node_weight: int = max_node_weight

        # global variables for data collecting
        self.write_data: bool = write_data
        self.__node_weight_sum: dict = {}
        self.__node_infected_sum: dict = {}
        self.__node_exposure_sum: dict = {}
        self.__agent_exposure_time: dict = {}

    def __get_buttons(self) -> None:
        """
        Get all the instances of a button to update the command with
        functionalisities within the class Simulation.
        Only used when self.visualize is True.
        """
        button_list = Button.get_all_instances()
        button_list['Pause'].set_command(self.__toggle_pause)
        return

    def __init_weight(self) -> None:
        """Initilize the weights of the graph to 0."""
        for node in self.graph.nodes:
            self.graph.nodes[node]['weight'] = 0
        return

    def __init_customer_distribution(self) -> list[int]:
        """
        Generates a list of arrival times for customers in a store.

        Returns:
        -------
        - List[int]: A list of arrival times for customers in a store.
        """
        arrival_times = np.random.uniform(0, self.timesteps,
                                          int(self.n_agents)).astype(int).tolist()

        return arrival_times

    def __init_agents(self) -> None:
        """Initialize agents with random characteristics and paths."""
        arrival_times = self.__init_customer_distribution()

        for i in range(self.n_agents):
            movement_speed = round(random.uniform(0, 1),
                                   Simulation.__ROUNDING_AGENT_ACTIONS)
            wait_time = round(random.uniform(0, 1),
                              Simulation.__ROUNDING_AGENT_ACTIONS)

            path, waypoints = path_generator.get_complete_route(
                self.graph, 0, 118).values()
            agent = Agent(id=i, arrival_time=arrival_times[i],
                          path=path, waypoints=waypoints,
                          movement_speed=movement_speed, wait_time=wait_time)
            AgentValidator(agent).validate()
            self.agents_list.append(agent)

        return

    def __init_infected(self) -> None:
        """Initialize a subset of agents as infected based on the initial infection rate."""
        infected_agents_count = int(self.infecting_init * self.n_agents)
        infected_agents_indices = random.sample(
            range(self.n_agents), infected_agents_count
        )
        Agent.set_init_infected_amount(len(infected_agents_indices))

        for index in infected_agents_indices:
            self.agents_list[index].is_infected = True
            self.agents_list[index].is_init_infected = True
        return

    def __init_dicts(self, dict_list: list[dict]):
        """
        Initialize dictionaries for data collection.

        Parameters:
        ---------
        - dict_list (list[dict]): A list of dicts to initialize.
        """
        for _, node in enumerate(self.graph.nodes):
            for dict in dict_list:
                dict[node] = 0

        for timestep in range(self.timesteps):
            self.__agent_exposure_time[timestep] = {}

            for agent in self.agents_list:
                self.__agent_exposure_time[timestep][agent.id] = 0
        return

    def __init_data_file(self) -> None:
        """
        Initialize a json file for the simulation to write the data to.
        Creates a new file if file doesn't exist or if there is a error.
        """
        self.data_dict = {"config": {
            "iterations": self.iterations,
            "timesteps": self.timesteps,
            "n_agents": self.n_agents,
            "infecting_adjacency": self.infecting_adjacency,
            "infecting_init": self.infecting_init,
            "infecting_prob": self.infecting_prob,
            "max_node_weight": self.max_node_weight
        }}
        standard_dict = {self.graph_type: self.data_dict}

        try:
            with open(Simulation.__JSON_FILE, "r") as file:
                existing_data = json.load(file)
                existing_data.update(standard_dict)
                standard_dict = existing_data
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass

        with open(Simulation.__JSON_FILE, "w") as file:
            json.dump(standard_dict, file, indent=Simulation.__JSON_INDENT)

        return

    def __get_neighbor_nodes(self, center_node: int) -> list[int]:
        """
        Get a list of neighboring nodes within a certain radius.

        Parameters:
        ---------
        - center_node (int): The node for which neighbors are sought.

        Returns:
        -------
        - list[int]: A list of neighboring nodes.
        """
        if center_node == None or center_node == 0:
            return []

        ego_graph = nx.ego_graph(
            self.graph, center_node, radius=self.infecting_adjacency)
        list_of_neighbors = [node for node in ego_graph.nodes()
                             if ego_graph.nodes[node]['weight'] > 0]

        if len(list_of_neighbors) > 0:
            return list_of_neighbors
        return []

    def __update_weights(self, agents: list[Agent], timestep: int) -> None:
        """
        Update weights of nodes based on agent positions at a given timestep.

        Parameters:
        ---------
        - agents (list[Agent]): List of Agent objects.
        - timestep (int): Current timestep of the simulation.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]['weight'] = 0

        for agent in agents:
            if timestep >= agent.arrival_time:
                if agent.currentNode is not None:
                    self.graph.nodes[agent.currentNode]['weight'] += 1

        for node, values in self.graph.nodes.items():
            weight = values['weight']
            self.__node_weight_sum[node] += weight
        return

    def __update(self, timestep: int) -> None:
        """
        Update the graph based on agent positions and node weights at a given timestep.

        Parameters:
        ---------
        - timestep (int): Current timestep of the simulation.
        """
        self.__update_weights(self.agents_list, timestep)
        self.__process_agents(timestep)

        self.agents_list = [
            agent for agent in self.agents_list
            if not (agent.currentIndex != -1 and agent.currentNode is None)
        ]

        for agent in self.agents_list:
            if timestep >= agent.arrival_time:
                agent.move(self.graph)

        if timestep == self.timesteps - 1:
            return

        if self.visualize:
            self.__draw(timestep)
        return

    def __process_agents(self, timestep: int) -> None:
        """
        Process agent infections and update infected node count.

        Parameters:
        ---------
        - timestep (int): Current timestep of the simulation.
        """
        for agent in self.agents_list:
            neighbor_list = self.__get_neighbor_nodes(agent.currentNode)
            for other_agent in self.agents_list:
                self.__process_update_infection(
                    agent, other_agent, neighbor_list)
            self.__process_update_exposure_time(agent, neighbor_list, timestep)
        return

    def __process_update_infection(self, agent: Agent, other_agent: Agent,
                                   neighbor_list: list[int]) -> None:
        """
        Process infection for the given agent and other_agent, and update infected node count.

        Parameters:
        ---------
        - agent (Agent): Current Agent.
        - other_agent (Agent): Another Agent being considered for infection.
        - neighbor_list (list[int]): List of neighboring nodes.
        """
        current_node = agent.currentNode
        if (other_agent.currentNode in neighbor_list
            and agent.id != other_agent.id
                and agent.is_infected):

            other_node = other_agent.currentNode

            if agent.infect(other_agent):
                # Update infected node count
                self.__node_infected_sum[other_node] += 1
            self.__node_exposure_sum[current_node] += 1
        return

    def __process_update_exposure_time(self, agent: Agent,
                                       neighbor_list: list[int], timestep: int) -> None:
        """
        Process and update exposure time for the given node.

        Parameters:
        ---------
        - agent (Agent): Current Agent.
        - neighbor_list (list[int]): List of neighboring nodes.
        - timestep (int): Current timestep of the simulation.
        """
        if agent.is_infected:
            return

        for neighbor_node in neighbor_list:
            neighbor_agent = next((a for a in self.agents_list if a.currentNode == neighbor_node
                                   and a.is_infected), None)

            if neighbor_agent is not None:
                if agent.id not in self.__agent_exposure_time[timestep]:
                    self.__agent_exposure_time[timestep][agent.id] = 1
                self.__agent_exposure_time[timestep][agent.id] += 1
        return

    def __draw(self, timestep: int) -> None:
        """
        Draw the new network on the frame.

        Parameters:
        ---------
        - timestep (int): Current timestep of the simulation.
        """
        self.axes.clear()
        color_map: list[str] = [f'#FF{int(np.clip(255 * (1 - data["weight"]), 0, 255)):02X}00'
                                for _, data in self.graph.nodes(data=True)]
        node_weights: dict = {node: round(data['weight'], Simulation.__ROUNDING_DECIMAL)
                              for node, data in self.graph.nodes(data=True)}

        nx.draw(self.graph, pos=self.pos, with_labels=True, node_color=color_map, labels=node_weights,
                cmap=plt.cm.Blues, vmin=0, vmax=max(color_map),
                node_size=700, font_size=8, edge_color='gray')

        self.axes.set_title(f'Timestep: {timestep + 1}')
        self.canvas.draw()
        return

    def simulate(self, amount: int) -> bool:
        """
        Run the simulation for the specified number of iterations.

        Parameters:
        ---------
        - amount (int): Number of iterations to run the simulation.

        Returns:
        --------
        - bool: True if the simulation completed successfully, False otherwise.
        """
        try:
            self.iterations = amount
            self.total_infected_list = []
            self.total_cases_list = []

            progress_bar = tqdm(total=self.iterations, desc="Processing")

            for _ in range(self.iterations):
                self.__simulate_controler()
                results = self.__get_simulation_infected()
                self.total_infected_list.append(results['total infected'])
                self.total_cases_list.append(results['total cases'])

                progress_bar.update(1)

            progress_bar.close()
            return True
        except Exception as e:
            print(f"Error during simulation: {e}")
            return False

    def __simulate_controler(self) -> None:
        """
        Internal method to control the simulation process.
        """
        Agent.set_infecting_prob(self.infecting_prob)
        Agent.set_max_node_weight(self.max_node_weight)
        self.__init_weight()
        self.__init_data_file()
        self.__init_agents()
        self.__init_infected()
        self.__init_dicts(
            [self.__node_weight_sum, self.__node_infected_sum, self.__node_exposure_sum]
        )

        if self.visualize:
            self.figure: Figure = self.frame.get_figure()
            self.animation: FuncAnimation = FuncAnimation(self.figure, self.__update,
                                                          frames=self.timesteps,
                                                          interval=self.interval, repeat=False)
            self.frame.run_root()
            self.frame.quit_root()
        else:
            for i in range(self.timesteps):
                self.__update(i)
        return

    def __toggle_pause(self) -> bool:
        """
        Toggle the pause state of the animation.

        Returns:
        ---------
        - bool: Current pause state (True if paused, False if resumed).
        """
        if not self.pause:
            print("Animation paused.")
            self.animation.event_source.stop()
            self.pause = True
        else:
            print("Animation resumed.")
            self.animation.event_source.start()
            self.pause = False
        return self.pause

    def __write_to_json_helper(self, data_source: str, return_dict: dict) -> None:
        """
        Add the collected data to the data_dict.

        Parameters:
        ---------
        - data_source (str): Name of the key for the data_dict.
        - return_dict (dict): The dict with the data that should be written
        """
        if self.write_data:
            self.data_dict[data_source] = return_dict
        return

    def __get_simulation_infected(self) -> tuple[int, int]:
        """
        Get the total number of infected agents and reset the counter.

        Returns:
        ---------
        - tuple[int, int]: Tuple containing total infected and total cases.
        """
        infected_tuple = Agent.get_total_infected()
        Agent.reset_infected()
        return infected_tuple

    def get_total_infected(self) -> dict[list[int], list[int]]:
        """
        Get a dict containing lists of total infected and total cases for each iteration.

        Returns:
        ---------
        - dict[list[int], list[int]]: Dict with lists of total infected and total cases.
        """
        return_dict = {
            'total infected': self.total_infected_list,
            'total cases': self.total_cases_list
        }

        self.__write_to_json_helper("get_total_infected", return_dict)
        return return_dict

    def get_average_infected(self) -> dict[float, float]:
        """
        Get a dict containing the average number of infected and average cases over all iterations.

        Returns:
        ---------
        - dict[float, float]: Dict with average infected and average cases.
        """
        return_dict = {
            'average infected': round(np.average(self.total_infected_list),
                                      Simulation.__ROUNDING_PERSON),
            'average cases': round(np.average(self.total_cases_list),
                                   Simulation.__ROUNDING_PERSON)
        }

        self.__write_to_json_helper("get_average_infected", return_dict)
        return return_dict

    def get_node_weight(self) -> dict[dict, dict]:
        """
        Get a dictionary containing average and total node weights.

        Returns:
        - dict[dict, dict]: Dictionary with average and total node weights.
        """
        n = self.timesteps * self.iterations
        average_weights = {node: round(weight / n, Simulation.__ROUNDING_DECIMAL)
                           for node, weight in self.__node_weight_sum.items()}

        return_dict = {
            'average': average_weights,
            'total': self.__node_weight_sum
        }

        self.__write_to_json_helper("get_node_weight", return_dict)
        return return_dict

    def get_node_infected(self) -> dict[dict, dict]:
        """
        Get a dictionary containing average and total infected node counts.

        Returns:
        - dict[dict, dict]: Dictionary with average and total infected node counts.
        """
        n = self.timesteps * self.iterations
        average_infected = {node: round(infected / n, Simulation.__ROUNDING_DECIMAL)
                            for node, infected in self.__node_infected_sum.items()}
        return_dict = {
            'average': average_infected,
            'total': self.__node_infected_sum
        }

        self.__write_to_json_helper("get_node_infected", return_dict)
        return return_dict

    def get_node_exposure(self) -> dict[dict, dict]:
        """
        Get a dictionary containing average and total node exposure counts.

        Returns:
        ---------
        - dict[dict, dict]: Dictionary with average and total node exposure counts.
        """
        n = self.timesteps * self.iterations
        average_exposure = {node: round(exposure / n, Simulation.__ROUNDING_DECIMAL)
                            for node, exposure in self.__node_exposure_sum.items()}
        return_dict = {
            'average': average_exposure,
            'total': self.__node_exposure_sum
        }

        self.__write_to_json_helper("get_node_exposure", return_dict)
        return return_dict

    def get_agent_exposure(self) -> dict[dict, dict]:
        """
        Get a dictionary containing average and total agent exposure times.

        Returns:
        ---------
        - dict[dict, dict]: Dictionary with average and total agent exposure times.
        """
        n = self.iterations
        sorted_exposure_time = {
            key1: {key2: value2
                   for key2, value2 in sorted(inner_dict.items(),
                                              key=lambda item: item[1], reverse=True)}
            for key1, inner_dict in sorted(self.__agent_exposure_time.items())
        }
        average_exposure_time = {key: {k: round(total / n, Simulation.__ROUNDING_DECIMAL)
                                       for k, total in inner_dict.items()}
                                 for key, inner_dict in sorted_exposure_time.items()}
        return_dict = {
            'average': average_exposure_time,
            'total': sorted_exposure_time
        }

        self.__write_to_json_helper("get_agent_exposure", return_dict)
        return return_dict

    def get_all_data(self) -> list[dict] | bool:
        """
        Get all the data at once in a list.
        Practicle use is to run it with write_json_data(),
        so that all the data is sorted at once.

        Returns:
        ---------
        - list[dict]: List with all the collected data as dictionary.
        - bool: False if the function failed.
        """
        try:
            return [
                self.get_node_exposure(),
                self.get_node_infected(),
                self.get_node_weight(),
                self.get_agent_exposure(),
                self.get_average_infected(),
                self.get_total_infected()
            ]
        except BaseException as error:
            print('An exception occurred: {}'.format(error))
            return False

    def write_json_data(self) -> bool:
        """
        Write the data to the JSON file.

        Returns:
        ---------
        - bool: Succes of the function (True if it succeed, False otherwise)
        """
        try:
            with open(Simulation.__JSON_FILE, "r") as file:
                existing_data = json.load(file)

            existing_data[self.graph_type] = self.data_dict

            with open(Simulation.__JSON_FILE, "w") as file:
                json.dump(existing_data, file, indent=Simulation.__JSON_INDENT)

            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False
