"""
validator.py

This script contains the Validator class and its subclasses for validating instances of other classes.
The Validator class is an abstract base class with a validate method that subclasses must implement.
Currently, there is a subclass for validating Agent instances.

Dependencies:
----
- tkinter
- networkx
- classes.agent
- classes.framework

Usage:
----
1. Import the required libraries: tkinter, networkx.
2. Import the Validator class and its subclasses from this file.
3. Create an instance of a Validator subclass and use its validate method to validate
    an instance of another class.

Example:
----
# Create an Agent instance
agent = Agent(1, 0, [1, 2, 3], [2], 1.0, 1.0)

# Create an AgentValidator instance
validator = AgentValidator(agent)

# Validate the agent
validator.validate()
"""

import tkinter as tk
import networkx as nx

from classes.agent import Agent
from classes.framework import Framework


class Validator:
    """
    Base class for validators. Subclasses must implement the validate method.
    """

    def __init__(self):
        pass

    def validate(self):
        """
        Abstract method to be implemented by subclasses. 
        This method is responsible for validating the attributes
        of the associated instance. Subclasses should provide concrete implementation
        to perform specific validation checks. If the validation fails,
        subclasses are expected to raise appropriate exceptions indicating the
        validation error. Returns True if the validator succeeds.

        Raises:
        ------
        - ValueError: If the parameter isn't valid.

        Returns:
        ------
        - bool: True is the class is valid.
        """
        raise NotImplementedError(
            "Subclasses must implement the validate method.")


class AgentValidator(Validator):
    """
    Validator class for Agent instances.

    Parameters
    ----------
    agent : Agent
        An instance of the Agent class to be validated.

    Methods
    -------
    validate()
        Validate the attributes of the associated Agent instance.
    validate_id()
        Validate the 'id' attribute of the associated Agent instance.
    validate_arrival_time()
        Validate the 'arrival_time' attribute of the associated Agent instance.
    validate_path()
        Validate the 'path' attribute of the associated Agent instance.
    validate_waypoints()
        Validate the 'waypoints' attribute of the associated Agent instance.
    validate_movement_speed()
        Validate the 'movement_speed' attribute of the associated Agent instance.
    validate_wait_time()
        Validate the 'wait_time' attribute of the associated Agent instance.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def validate(self) -> bool:
        self.validate_id()
        self.validate_arrival_time()
        self.validate_path()
        self.validate_waypoints()
        self.validate_movement_speed()
        self.validate_wait_time()
        return True

    def validate_id(self):
        if not isinstance(self.agent.id, int) or self.agent.id < 0:
            raise ValueError("Agent id must be a non-negative integer.")

    def validate_arrival_time(self):
        if not isinstance(self.agent.arrival_time, int) or self.agent.arrival_time < 0:
            raise ValueError(
                "Agent arrival time must be a non-negative integer.")

    def validate_path(self):
        if (not isinstance(self.agent.path, list) 
            or not all(isinstance(node, int) for node in self.agent.path)):
            raise ValueError("Agent path must be a list of integers.")

    def validate_waypoints(self):
        if (not isinstance(self.agent.waypoints, list)
                or not all(isinstance(node, int) for node in self.agent.waypoints)):
            raise ValueError("Agent waypoints must be a list of integers.")

    def validate_movement_speed(self):
        if (not isinstance(self.agent.movement_speed, (int, float))
                or not (0 <= self.agent.movement_speed <= 1)):
            raise ValueError(
                "Agent movement speed must be a number between 0 and 1.")

    def validate_wait_time(self):
        if (not isinstance(self.agent.wait_time, (int, float))
                or not (0 <= self.agent.wait_time <= 1)):
            raise ValueError(
                "Agent movement speed must be a number between 0 and 1.")


class ButtonValidator(Validator):
    """
    Validator class for Button instances.

    Parameters
    ----------
    button : Button
        An instance of the Button class to be validated.

    Methods
    -------
    validate()
        Validate the attributes of the associated Button instance.
    validate_container()
        Validate the 'container' attribute of the associated Button instance.
    validate_text()
        Validate the 'text' attribute of the associated Button instance.
    validate_image()
        Validate the 'image' attribute of the associated Button instance.
    validate_command()
        Validate the 'command' attribute of the associated Button instance.
    validate_font()
        Validate the 'font' attribute of the associated Button instance.
    """

    def __init__(self, button):
        super().__init__()
        self.button = button

    def validate(self) -> bool:
        self.validate_container()
        self.validate_text()
        self.validate_image()
        self.validate_command()
        self.validate_font()
        return True

    def validate_container(self):
        if not isinstance(self.button.container, tk.Frame):
            raise ValueError(
                "Button container must be a valid Tkinter widget.")

    def validate_text(self):
        if not isinstance(self.button.text, str):
            raise ValueError("Button text must be a string.")

    def validate_image(self):
        pass

    def validate_command(self):
        if not callable(self.button.command):
            raise ValueError("Button command must be a callable function.")

    def validate_font(self):
        if self.button.font is not None and not isinstance(self.button.font, dict):
            raise ValueError("Button font must be a dictionary.")


class FrameworkValidator(Validator):
    """
    Validator class for Framework instances.

    Parameters
    ----------
    framework : Framework
        An instance of the Framework class to be validated.

    Methods
    -------
    validate()
        Validate the attributes of the associated Framework instance.
    validate_title()
        Validate the 'title' attribute of the associated Framework instance.
    validate_height()
        Validate the 'height' attribute of the associated Framework instance.
    validate_width()
        Validate the 'width' attribute of the associated Framework instance.
    validate_fullscreen()
        Validate the 'fullscreen' attribute of the associated Framework instance.
    """

    def __init__(self, framework):
        super().__init__()
        self.framework = framework

    def validate(self) -> bool:
        self.validate_title()
        self.validate_height()
        self.validate_width()
        self.validate_fullscreen()
        return True

    def validate_title(self):
        if not isinstance(self.framework.title, str):
            raise ValueError("Framework title must be a string.")

    def validate_height(self):
        if not isinstance(self.framework.height, int) or self.framework.height <= 0:
            raise ValueError("Framework height must be a positive integer.")

    def validate_width(self):
        if not isinstance(self.framework.width, int) or self.framework.width <= 0:
            raise ValueError("Framework width must be a positive integer.")

    def validate_fullscreen(self):
        if not isinstance(self.framework.fullscreen, bool):
            raise ValueError("Framework fullscreen must be a boolean.")


class SimulationValidator(Validator):
    """
    Validator class for Simulation instances.

    Parameters
    ----------
    - simulation : Simulation
        An instance of the Simulation class to be validated.

    Methods
    -------
    validate()
        Validate the attributes of the associated Simulation instance.
    validate_graph()
        Validate the 'graph' attribute of the associated Simulation instance.
    validate_pos()
        Validate the 'pos' attribute of the associated Simulation instance.
    validate_timesteps()
        Validate the 'timesteps' attribute of the associated Simulation instance.
    validate_agents()
        Validate the 'agents_list' attribute of the associated Simulation instance.
    validate_interval()
        Validate the 'interval' attribute of the associated Simulation instance.
    validate_write_data()
        Validate the 'write_data' attribute of the associated Simulation instance.
    validate_visualize()
        Validate the 'visualize' attribute of the associated Simulation instance.
    validate_frame()
        Validate the 'frame' attribute of the associated Simulation instance.
    validate_iterations()
        Validate the 'iterations' attribute of the associated Simulation instance.
    validate_n_agent()
        Validate the 'n_agents' attribute of the associated Simulation instance.
    validate_infecting_prob()
        Validate the 'infecting_prob' attribute of the associated Simulation instance.
    validate_infecting_init()
        Validate the 'infecting_init' attribute of the associated Simulation instance.
    validate_infecting_adjacency()
        Validate the 'infecting_adjacency' attribute of the associated Simulation instance.
    validate_max_node_weight()
        Validate the 'max_node_weight' attribute of the associated Simulation instance.
    """

    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def validate(self) -> bool:
        self.validate_graph()
        self.validate_pos()
        self.validate_timesteps()
        self.validate_agents()
        self.validate_interval()
        self.validate_write_data()
        self.validate_visualize()
        self.validate_frame()
        self.validate_iterations()
        self.validate_n_agent()
        self.validate_infecting_prob()
        self.validate_infecting_init()
        self.validate_infecting_adjacency()
        self.validate_max_node_weight()
        return True

    def validate_graph(self):
        if not isinstance(self.simulation.graph, nx.Graph):
            raise ValueError(
                "Simulation graph must be a valid NetworkX graph.")

    def validate_pos(self):
        if not isinstance(self.simulation.pos, dict):
            raise ValueError("Simulation pos must be a dictionary.")

    def validate_timesteps(self):
        if not isinstance(self.simulation.timesteps, int) or self.simulation.timesteps <= 0:
            raise ValueError(
                "Simulation timesteps must be a positive integer.")

    def validate_agents(self):
        if (not isinstance(self.simulation.agents_list, list)
                or not all(isinstance(agent, Agent) for agent in self.simulation.agents_list)):
            raise ValueError(
                "Simulation agents must be a list of Agent instances.")

    def validate_interval(self):
        if not isinstance(self.simulation.interval, int) or self.simulation.interval <= 0:
            raise ValueError("Simulation interval must be a positive integer.")

    def validate_write_data(self):
        if not isinstance(self.simulation.write_data, bool):
            raise ValueError("Simulation write_data must be a boolean.")

    def validate_visualize(self):
        if not isinstance(self.simulation.visualize, bool):
            raise ValueError("Simulation visualize must be a boolean.")

    def validate_frame(self):
        if self.simulation.visualize and not isinstance(self.simulation.frame, Framework):
            raise ValueError(
                "Simulation frame must be a valid Framework instance when visualize is True.")

    def validate_iterations(self):
        if not isinstance(self.simulation.iterations, int) or self.simulation.iterations <= 0:
            raise ValueError(
                "Simulation iterations must be a positive integer.")

    def validate_n_agent(self):
        if not isinstance(self.simulation.n_agents, int) or self.simulation.n_agents <= 0:
            raise ValueError("Simulation n_agent must be a positive integer.")

    def validate_infecting_prob(self):
        if (not isinstance(self.simulation.infecting_prob, float)
                or not (0 <= self.simulation.infecting_prob <= 1)):
            raise ValueError(
                "Simulation infecting_prob must be a float between 0 and 1.")

    def validate_infecting_init(self):
        if (not isinstance(self.simulation.infecting_init, float)
                or not (0 <= self.simulation.infecting_init <= 1)):
            raise ValueError(
                "Simulation infecting_init must be a float between 0 and 1.")

    def validate_infecting_adjacency(self):
        if (not isinstance(self.simulation.infecting_adjacency, int)
                or self.simulation.infecting_adjacency < 0):
            raise ValueError(
                "Simulation infecting_adjacency must be a non-negative integer.")

    def validate_max_node_weight(self):
        if (not isinstance(self.simulation.max_node_weight, int)
                or self.simulation.max_node_weight <= 0):
            raise ValueError(
                "Simulation max_node_weight must be a positive integer.")
