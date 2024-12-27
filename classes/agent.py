"""
agent.py

This script contains the Agent class which represents an agent in a simulation.
The class includes methods for moving the agent through a network graph,
infecting the agent, and checking the agent's infection status.

Dependencies:
----
- random
- numpy

Usage:
----
1. Import the required libraries: random, numpy.
2. Import the Agent class from this file.
3. Create an instance of the Agent class and use its methods to simulate an agent's
movement and infection status.

Example:
----
# Create an Agent instance
agent = Agent(1, 0, [1, 2, 3], [2], 1.0, 1.0)

# Move the agent
agent.move()

# Infect the agent
agent.infect()

# Check the agent's infection status
print(agent.is_infected)
"""

import random
import numpy as np

from typing import Optional


class Agent:
    """
    Class representing an agent in a simulation.

    Parameters
    ----------
    id : int
        Unique identifier for the agent.
    arrival_time : int
        Time of the agent's arrival in the simulation.
    path : list[int]
        List specifying the agent's movement through nodes.
    waypoints : list[int]
        List of nodes where the agent may pause during movement.
    movement_speed : float, optional (default=1.0)
        Speed of agent movement.
        Must be between 0 and 1.
    wait_time : float, optional (default=1.0)
        Probability of the agent waiting at waypoints during movement.
        Must be between 0 and 1.

    Attributes
    ----------
    id : int
        Unique identifier for the agent.
    movement_speed : float
        Speed of agent movement.
    wait_time : float
        Probability of the agent waiting at waypoints during movement.
    currentNode : Optional[int]
        Current node the agent is located at.
    currentIndex : int
        Index indicating the agent's position in the path.
    path : list[int]
        List specifying the agent's movement through nodes.
    waypoints : list[int]
        List of nodes where the agent may pause during movement.
    arrivalTime : int
        Time of the agent's arrival in the simulation.
    is_infected : bool
        Indicates whether the agent is infected.
    is_init_infected : bool
        Indicates whether the agent is initially infected.

    Methods
    -------
    move(graph: nx.Graph) -> None
        Move the agent to the next node in its path or remove if at the end of the graph.
    infect(other: Agent) -> bool
        Attempt to infect another agent based on infection probability.
    """
    __infecting_prob: float = 0
    __total_infected: int = 0
    __total_init_infected: int = 0
    __total_cases: int = 0
    __max_node_weight: int

    @classmethod
    def increment_infected(cls, amount=1) -> None:
        """
        Increment the total number of infected agents.

        Parameters:
        ---------
        - amount (int): The number by which to increment the total infected count.
        """
        cls.__total_infected += amount
        return

    @classmethod
    def reset_infected(cls) -> None:
        """
        Reset the total infected, total cases, and total initial infected counts.
        This is used when using multiple iterations within the simulation.
        """
        cls.__total_infected = 0
        cls.__total_cases = 0
        cls.__total_init_infected = 0
        return

    @classmethod
    def set_max_node_weight(cls, weight=1) -> None:
        """
        Set the maximum node weight for agent movement.

        Parameters:
        ---------
        - weight (int): The maximum node weight for agent movement. Defaults to 1.
        """
        cls.__max_node_weight = weight
        return

    @classmethod
    def set_init_infected_amount(cls, amount=1) -> None:
        """
        Set the total initial infected count.

        Parameters:
        ---------
        - amount (int): The total initial infected count. Defaults to 1.
        """
        cls.__total_init_infected += amount
        return

    @classmethod
    def set_infecting_prob(cls, infecting_prob) -> None:
        """
        Set the infection probability for agent interactions.

        Parameters:
        ---------
        - infecting_prob (float): The infection probability. Must be between 0 and 1.
        """
        if 0 <= infecting_prob <= 1:
            cls.__infecting_prob = infecting_prob
        else:
            raise ValueError("Infecting probability must be between 0 and 1.")
        return

    @classmethod
    def get_total_infected(cls) -> tuple[int, int]:
        """
        Get a tuple with the total infected and total cases counts.

        Returns:
        ---------
        - Tuple[int, int]: A tuple containing the total infected and total cases counts.
        """
        cls.__total_cases = cls.__total_infected + cls.__total_init_infected
        return_dict = {
            'total infected': cls.__total_infected,
            'total cases': cls.__total_cases
        }
        return return_dict

    def __init__(self, id, arrival_time: int, path: list[int],
                 waypoints: list[int], movement_speed=1.0, wait_time=1.0) -> None:
        """
        Initialize an Agent object.

        Parameters:
        ---------
        - id (int): Unique identifier for the agent.
        - arrival_time (int): Time of the agent's arrival in the simulation.
        - path (List[int]): List specifying the agent's movement through nodes.
        - waypoints (List[int]): List of nodes where the agent may pause during movement.
        - movement_speed (float, optional): Speed of agent movement. Must be between 0 and 1. Default: 1.0.
        - wait_time (float, optional): Probability of the agent waiting at waypoints during movement.
            Must be between 0 and 1. Defaults: 1.0.
        """
        self.id = id
        self.currentNode: Optional[int] = None
        self.currentIndex: int = -1

        self.movement_speed: float = movement_speed
        self.wait_time: float = wait_time
        self.path: list[int] = path
        self.waypoints: list[int] = waypoints
        self.arrival_time: int = arrival_time

        self.is_infected: bool = False
        self.is_init_infected: bool = False

    def move(self, graph) -> None:
        """
        Move the agent to the next node in its path or remove if at the end of the graph.

        Parameters:
        ---------
        - graph (nx.Graph): The graph representing the network.
        """
        def get_next_node():
            """Get the next node in the path."""
            next_index = self.currentIndex + 1
            return self.path[next_index] if next_index < len(self.path) else self.path[-1]

        def move_to_next_node():
            """Move agent to the next node."""
            self.currentIndex += 1
            self.currentNode = self.path[self.currentIndex]

        next_node = get_next_node()

        if self.currentIndex < len(self.path) - 1:
            if (
                self.currentNode in self.waypoints
                and random.random() < 1 - self.wait_time
                and graph.nodes[next_node]['weight'] <= Agent.__max_node_weight
            ):
                move_to_next_node()
            elif random.random() < self.movement_speed or self.currentIndex == -1:
                move_to_next_node()
        else:
            # If the agent is at the last, remove them from the graph
            self.currentNode = None
        return

    def infect(self, other: "Agent") -> bool:
        """
        Attempt to infect another agent based on infection probability.

        Parameters:
        ---------
        - other (Agent): The other agent to attempt infection.

        Returns:
        ---------
        - bool: True if the infection is successful, False otherwise.
        """
        if self.is_init_infected and not other.is_infected:
            if np.random.uniform() <= self.__infecting_prob:
                other.is_infected = True
                Agent.increment_infected()
                return True
        return False

    def __str__(self):
        """
        Return a string representation of the Agent object.

        Returns:
        ---------
        - str: String representation of the Agent object.
        """
        return f"Agent object: {self.id}"
