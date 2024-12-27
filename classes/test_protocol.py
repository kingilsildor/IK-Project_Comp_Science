"""
test_protocol.py

This script contains unit tests for the Simulation class in the simulation file. 
The tests ensure the correctness and functionality of the simulation's data collection methods.

Dependencies:
----
- unittest
- tkinter
- networkx

Usage:
----
1. Import the required libraries: unittest, networkx.
2. Import the Simulation class and other necessary classes from the simulation file.
3. Run the unit tests using the `unittest` module.

Example:
----
class TestSimulationFunctions(unittest.TestCase):
    # ... (unit tests go here)
"""
import unittest
import tkinter as tk
import networkx as nx
import generators.generate_graph as graph_generator
import generators.generate_paths as path_generator

from classes.agent import Agent
from classes.framework import Button, Framework
from classes.simulation import Simulation
from classes.validator import AgentValidator, ButtonValidator, FrameworkValidator, SimulationValidator

TIMESTEPS = 10


class TestGenerateGraph(unittest.TestCase):
    """
    Test class for generating graphs.

    Tests:
    ----
    - test_graph_similarity(): Checks if two undirected graphs have the same nodes, edges, and sets.
    - test_node_amount(): Verifies the expected number of nodes in undirected and directed graphs (116).
    """

    def test_graph_similarity(self):
        G = graph_generator.generate_graph_undirected()
        G2 = graph_generator.generate_graph_undirected()

        self.assertEqual(len(G.nodes), len(G2.nodes))
        self.assertEqual(len(G.edges), len(G2.edges))
        self.assertEqual(G.nodes, G2.nodes)
        self.assertEqual(G.edges, G2.edges)

    def test_node_amount(self):
        G = graph_generator.generate_graph_undirected()
        G2 = graph_generator.generate_graph_directed()
        self.assertEqual(len(G.nodes), 116)
        self.assertEqual(len(G2.nodes), 116)


class TestGeneratePaths(unittest.TestCase):
    """
    Test class for generating paths in an undirected graph.

    Tests:
    ----
    - test_generate_paths(): Tests the generation of complete paths with specific criteria.
    """

    def test_generate_paths(self):
        G = graph_generator.generate_graph_undirected()
        for _ in range(10):
            route = path_generator.get_complete_route(G, 0, 118)
            route = route['route']
            assert route[0] == 0
            assert route[-1] == 118
            assert len(route) > 2


class TestSimulation(unittest.TestCase):
    """
    Test class for the simulation process and visualization.

    Tests:
    ----
    - test_simulation_initialization(): Checks if a Simulation object is initialized correctly.
    - test_simulation_simulation_process(): Tests the simulation process for a single iteration.
    - test_simulation_visualization(): Verifies successful creation of Framework and Simulation objects with visualization.
    - test_simulation_visualization_error(): Ensures a ValueError is raised for invalid visualization parameters.
    """

    def setUp(self):
        self.graph = graph_generator.generate_graph_undirected()
        self.pos = graph_generator.generate_pos(self.graph)

    def test_simulation_initialization(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        self.assertIsInstance(simulation, Simulation)
        self.assertEqual(simulation.graph, self.graph)
        self.assertEqual(simulation.pos, self.pos)

    def test_simulation_simulation_process(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        self.assertTrue(simulation.simulate(amount=1))

    def test_simulation_visualization(self):
        framework = Framework("unit_test", 500, 500)
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS,
                                visualize=True, frame=framework)
        self.assertIsInstance(framework, Framework)
        self.assertIsInstance(simulation, Simulation)

    def test_simulation_visualization_error(self):
        with self.assertRaises(ValueError):
            Simulation(self.graph, self.pos, timesteps=TIMESTEPS,
                       visualize=True, frame=None)


class TestSimulationFunctions(unittest.TestCase):
    """
    Test class for simulation-related functions and statistics.

    Tests:
    ----
    - A set of functions related to obtaining statistics from the simulation results.
    """

    def setUp(self):
        self.graph = graph_generator.generate_graph_undirected()
        self.pos = graph_generator.generate_pos(self.graph)

    def test_get_node_exposure(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_node_exposure()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('average', result)
        self.assertIn('total', result)
        self.assertIsInstance(result['average'], dict)
        self.assertIsInstance(result['total'], dict)

    def test_get_node_infected(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_node_infected()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('average', result)
        self.assertIn('total', result)
        self.assertIsInstance(result['average'], dict)
        self.assertIsInstance(result['total'], dict)

    def test_get_node_weight(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_node_weight()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('average', result)
        self.assertIn('total', result)
        self.assertIsInstance(result['average'], dict)
        self.assertIsInstance(result['total'], dict)

    def test_get_agent_exposure(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_agent_exposure()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('average', result)
        self.assertIn('total', result)
        self.assertIsInstance(result['average'], dict)
        self.assertIsInstance(result['total'], dict)

    def test_get_average_infected(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_average_infected()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('average infected', result)
        self.assertIn('average cases', result)
        self.assertIsInstance(result['average infected'], float)
        self.assertIsInstance(result['average cases'], float)

    def test_get_total_infected(self):
        simulation = Simulation(self.graph, self.pos, timesteps=TIMESTEPS)
        simulation.simulate(amount=1)
        result = simulation.get_total_infected()

        self.assertIsInstance(result, dict)
        self.assertTrue(result)

        self.assertIn('total infected', result)
        self.assertIn('total cases', result)
        self.assertIsInstance(result['total infected'], list)
        self.assertIsInstance(result['total cases'], list)


class TestAgentValidator(unittest.TestCase):
    """
    Test class for validating Agent objects.

    Tests:
    ----
    - test_valid_agent(): Validates an Agent object with valid attributes.
    - test_invalid_agent(): Ensures a ValueError is raised for an Agent object with invalid attributes.
    """

    def test_valid_agent(self):
        agent = Agent(id=1, arrival_time=0, path=[
                      1, 2, 3], waypoints=[4, 5, 6])
        agent_validator = AgentValidator(agent)
        agent_validator.validate()

    def test_invalid_agent(self):
        with self.assertRaises(ValueError):
            agent = Agent(id=-1, arrival_time='invalid',
                          path=[1, 2, 3], waypoints=[4, 5, 6])
            agent_validator = AgentValidator(agent)
            agent_validator.validate()


class TestButtonValidator(unittest.TestCase):
    """
    Test class for validating Button objects.

    Tests:
    ----
    - test_valid_button(): Validates a Button object with valid attributes.
    - test_invalid_button(): Ensures a ValueError is raised for a Button object with invalid attributes.
    """

    def test_valid_button(self):
        root = tk.Tk()
        button = Button(container=tk.Frame(root), text="Click Me",
                        command=lambda: print("Button Clicked"))
        button_validator = ButtonValidator(button)
        button_validator.validate()

    def test_invalid_button(self):
        with self.assertRaises(ValueError):
            button = Button(container=None, text=123,
                            command="invalid_command")
            button_validator = ButtonValidator(button)
            button_validator.validate()


class TestFrameworkValidator(unittest.TestCase):
    """
    Test class for validating Framework objects.

    Tests:
    - test_valid_framework(): Validates a Framework object with valid attributes.
    - test_invalid_framework(): Ensures a Tkinter TclError is raised for a Framework object with invalid attributes.
    """

    def test_valid_framework(self):
        framework = Framework(title="My Framework", height=500, width=800)
        framework_validator = FrameworkValidator(framework)
        framework_validator.validate()

    def test_invalid_framework(self):
        with self.assertRaises(tk.TclError):
            framework = Framework(title=123, height='invalid', width=-800)
            framework_validator = FrameworkValidator(framework)
            framework_validator.validate()


class TestSimulationValidator(unittest.TestCase):
    """
    Test class for validating Simulation objects.

    Tests:
    - test_valid_simulation(): Validates a Simulation object with valid attributes.
    - test_invalid_simulation(): Ensures a ValueError is raised for a Simulation object with invalid attributes.
    """

    def test_valid_simulation(self):
        graph = nx.Graph()
        pos = {}
        agent = Agent(id=1, arrival_time=0, path=[
                      1, 2, 3], waypoints=[4, 5, 6])
        simulation = Simulation(graph=graph, pos=pos,
                                timesteps=100, agents=[agent])
        simulation_validator = SimulationValidator(simulation)
        simulation_validator.validate()

    def test_invalid_simulation(self):
        with self.assertRaises(ValueError):
            graph = nx.Graph()
            pos = {}
            agent = Agent(id=1, arrival_time=0, path=[
                          1, 2, 3], waypoints=[4, 5, 6])
            simulation = Simulation(
                graph=graph, pos=pos, timesteps=-100, agents=[agent])
            simulation_validator = SimulationValidator(simulation)
            simulation_validator.validate()


if __name__ == '__main__':
    unittest.main()
