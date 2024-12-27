# Social Distancing


## Description

This project is a simulation framework that utilizes agents to perform computational tasks. It provides a flexible and extensible architecture for running simulations and validating the results. The project includes various generators for creating graphs, map plots, random paths, and other plots. It also includes classes for agents, the simulation framework, and validation. The project uses a JSON file for simulation data.

In the face of global challenges posed by infectious diseases, understanding the impact of preventive measures becomes important for public health. One such preventive strategy that gained considerable attention during the COVID-19 pandemic, is the dedicated walking paths within supermarkets and other stores (Shumsky et al, 2021; Tsukanov et al, 2021). The change in walking path entails to minimize close physical interactions among individuals, to curb the transmission of infectious diseases (Ying & O’Clery, 2021). This research seeks to delve into the critical question; “To what extent do the dedicated walking paths minimize the probability of getting infected by Covid-19 in a supermarket?”. By simulating the dynamics of disease transmission within a store, we aim to unravel the effect of customer flow on the spread of infectious diseases such as COVID-19.

The main inspiration for this project comes from the paper modeling COVID-19 transmission in supermarkets using an agent-based model (Ying & O’Clery, 2021). This paper utilizes a network based approach to measure the effectiveness of a dedicated walking path on the spread of COVID-19. Our approach differs from this paper, but it contains some useful insights which will apply to our model. Within this research the focus is more on the comparison between with dedicated walking paths and without.


## Table of Contents

- [Getting Started](#getting-started)
- [Possible Issues](#possible-issues)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributors](#contributors)
- [References](#references)

## Getting Started

1. Clone the repository.
2. Install the required dependencies from `requirements.txt`. It's best to run it in python 3.11.4.
3. Navigate to `main.py`.
4. Read the documentation.
5. Change the config if nessecary.
6. Run `main.py`.

After running the main a plot will be shown for running the code review.
The current parameters in the config aren't the one used for the simulation data as shown in `__data_used_for_plots.json`, but used to quickly run a simulation.

Depening on the hardware a simulation of 30 iterations with 600 agents and 1000 timesteps, can take between 2 and 5 hours. Use the plot functions with the given data if you just want to see the figures.

## Possible issues
Depending on the hardware, SAlib can't be found.
This is important to know if you want to run the sensitivity analysis.
The sensitivity analysis was done on a Macbook with Python version 3.9.6, whereby the other code was writen in 3.11.4.
We didn't expect to have problems with this, but sadly here we are.

If you want to run the sensitivity analysis you have to keep this in mind.

## Usage

`main.py` is the main entry point for the simulation project. It simulates the impact of dedicated walking paths on COVID-19 transmission in a supermarket using an agent-based approach. The simulation parameters include the supermarket framework, graph generation, path generation, agent details, and simulation dynamics. The script also includes a comparison of scenarios with and without dedicated walking paths.

The script contains several functions:
* `get_json()`: Gives an example on how to exces the JSON file.
* `run_unit_test()`: Runs the unit tests defined in test_protocol.py.
* `run_sensitivity_analysis()`: Runs the sensitivity analysis in sensitivity_analysis.py
* `run_simulation_with_visualization(ITERATIONS, graph)`: Runs the simulation with a visualization. It generates a framework, creates a simulation, validates the framework and simulation, runs the simulation, retrieves all data, and writes the data to a JSON file.
* `run_simulation(ITERATIONS, graph)`: Similar to the previous function but without visualization.
* `main()`: The main function that sets the number of iterations and generates directed and undirected graphs for the simulation. It then runs the simulation for both types of graphs. Currently within the `main()` the simulation with visualization is commented

The script ends by calling the `main()` function and running the unit tests.

```python
def run_simulation(graph, config, visualization=False):
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
        # Params changed to run a quicker simulation.
        "iterations": 1,
        "timesteps": 10,
        "n_agents": 5,
        # Params that doesn't change.
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
    run_sensitivity_analysis()
```

within the `main.py` you can call the plots, unit test and sensitivity analysis.

## File Descriptions

- `classes/agent.py`: Defines the agent class used in the simulation.
- `classes/framework.py`: Defines the framework for the simulation.
- `classes/simulation.py`: Defines the simulation class.
- `classes/validator.py`: Defines the validator class for validating simulation results.
- `classes/test_protocol.py`: Contains tests for the simulation protocol.
- `data/simulation_data.json`: Contains data used in the simulation.
- `generators/generate_graph.py`: Generates graphs for the simulation.
- `generators/generate_plots.py`: Generates various plots for the simulation.
- `generators/generate_paths.py`: Generates paths for the simulation.
- `main.py`: The main entry point for the simulation.
- `sensitivity_analysis.py`: Contains tests for sensitivity analysis.

## Contributors

We planed on collaborating in most parts of the assignment. We made a Kanban board to structure the tasks at hand, divide them into small tickets, and distribute the workload evenly. For the presentation and documentation, we have collaborated on all parts through longer meetings.


- **Mees**: Worked on making realistic human movement in store. Made code for visualizing the results and built the necesarry unit tests for it. Also added the visualizations to the poster.

- **Tycho**: Worked on creating the framework and all the classes, preparing the data for analysis, writing documentation and the test cases.

- **Thomas**: Built custom pathfinding algorithm based on existing shopping carts to model human-like movement. Built and performed the sensitivity analysis. Performed elementary data analysis.

The old branches are deleted to create a clean gitlab page.
All the commits can be found in main.


## References
Shumsky, R. A., Debo, L., Lebeaux, R. M., Nguyen, Q. P., & Hoen, A. G. (2021). Retail store customer flow and COVID-19 transmission. *Proceedings of the National Academy of Sciences, 118*(11), e2019225118.

Tsukanov, A. A., Senjkevich, A. M., Fedorov, M. V., & Brilliantov, N. V. (2021). How risky is it to visit a supermarket during the pandemic?. *Plos one, 16*(7), e0253835.

Ying, F., & O’Clery, N. (2021). Modelling COVID-19 transmission in supermarkets using an agent-based model. *Plos one, 16*(4), e0249821.
