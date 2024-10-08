# ### Research Coding Task: DCOPs Benchmark and Algorithm Evaluation
# **Objective:** Implement a simulation of Distributed Constraint Optimization Problems (DCOPs) and evaluate the performance of two algorithms, DSA and MGM, over multiple instances.
# #### Task Breakdown:
# 1. **Create a Random DCOPs Benchmark:**
#    - **Graph Construction:**
#      - **Agents and Edges:**
#        - Create a fully connected graph with **10 agents** (nodes).
#        - Add edges randomly while ensuring the graph's edge density is around **20%**. Edge density is calculated as the ratio of the number of edges to the maximum possible number of edges in the graph.
#        - For a fully connected graph of 10 nodes, there are a total of \(\frac{n(n-1)}{2}\) edges, where \(n\) is the number of nodes.
#        - Use a random function to assign a cost to each edge. The cost should be a random value within a reasonable range (e.g., between 1 and 10).
#    - **Python Implementation:**
#      - Use the **NetworkX** library to create and manipulate the graph.
# 2. **Implement DCOP Algorithms:**
#    - **DSA (Distributed Stochastic Algorithm):**
#      - Implement the DSA algorithm to solve the DCOP. This algorithm involves agents making local decisions to minimize the overall cost based on stochastic approaches.
#    - **MGM (Maximum Gain Method):**
#      - Implement the MGM algorithm. This approach focuses on selecting actions that maximize the gain (or reduction in cost) locally for each agent.
# 3. **Generate Instances and Run Simulations:**
#    - **Instance Generation:**
#      - Generate **10 different instances** of the DCOP benchmark with different random edge costs and structures.
#    - **Algorithm Execution:**
#      - For each instance, run both algorithms (DSA and MGM) for **1000 iterations**.
#      - Track and record the cost after each iteration to observe the algorithms’ performance over time.
# 4. **Performance Analysis:**
#    - **Data Collection:**
#      - Collect the average cost of the solution after each iteration for both algorithms.
#      - For each algorithm, average the costs over all 10 instances to obtain a smoothed performance curve.
#    - **Graph Creation:**
#      - Plot the results using **matplotlib** or a similar plotting library.
#      - The X-axis should represent the number of iterations (from 1 to 1000).
#      - The Y-axis should represent the average cost of the solution.
#      - Plot two lines on the graph: one for DSA and one for MGM to compare their performance.
### Example Code Outline:

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_random_dcop(num_agents, edge_density, graph_type):
    if graph_type == 'complete':
        G = nx.complete_graph(num_agents)
    elif graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_agents, 2)

    edges = list(G.edges())
    num_edges = len(edges)
    num_keep_edges = int(num_edges * edge_density)
    # Randomly sample edges to keep
    keep_edges = np.random.choice(num_edges, num_keep_edges, replace=False)
    keep_edges = [edges[idx] for idx in keep_edges]
    # keep_edges = [(u, v) for u, v in edges if u in keep_edges]
    G.remove_edges_from(edges)
    G.add_edges_from(keep_edges)
    # Assign random costs to edges
    # for u, v in G.edges():
    #     G[u][v]['cost'] = np.random.randint(1, 11)  # Random cost between 1 and 10
    return G

def create_cost_tables(G, num_actions, cost_range=101):
    # Each edge has a cost table based on the possible actions of the agents (both agents have actions between 0-num_actions, costing between 1- 100)
    for u, v in G.edges():
        G[u][v]['cost_table'] = np.random.randint(1, cost_range, size=(num_actions, num_actions))

def assign_initial_actions(G, num_actions):
    # Randomly assign initial actions to agents
    for agent in G.nodes():
        G.nodes[agent]['action'] = np.random.randint(0, num_actions)

def run_dsa_algorithm(G, num_iterations, num_actions, threshold):
    # Implement DSA algorithm here
    average_costs = []
    for i in range(num_iterations):
        for agent in G.nodes():
            agent_action = G.nodes[agent]['action']
            # action_gains = {0: 0, 1: 0, 2: 0}
            action_gains = {i: 0 for i in range(num_actions)}
            optimal_action = agent_action
            for change_action in range(num_actions):
                max_gain = 0
                for neighbor in G.neighbors(agent):
                    # receive information from neighbors
                    neighbor_action = G.nodes[neighbor]['action']
                    # calculate the maximum decrease in cost if action is changed
                    cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                    change_action_cost = G[agent][neighbor]['cost_table'][change_action][neighbor_action]
                    action_gains[change_action] +=  cur_cost - change_action_cost

                if action_gains[change_action] > max_gain:
                    max_gain = action_gains[change_action]
                    optimal_action = change_action
            
            #choose random index of action with threshold = probability of choosing the optimal action
            
            # threshold = 0.95
            p = [(1 - threshold) / (num_actions - 1) if i != optimal_action else threshold for i in range(num_actions)]
            optimal_action = np.random.choice([i for i in range(num_actions)], p=p)
            G.nodes[agent]['action'] = optimal_action
        # Calculate the avg cost per iteration
        avg_cost = 0
        for u,v in G.edges():
            avg_cost += G[u][v]['cost_table'][G.nodes[u]['action']][G.nodes[v]['action']]
        avg_cost = avg_cost / len(G.edges())
        average_costs.append(avg_cost)

    return average_costs

def run_dsan_algorithm(G, num_iterations, num_actions, temperature):
    # Implement DSA algorithm here
    average_costs = []
    for i in range(num_iterations):
        for agent in G.nodes():
            probs = []
            agent_action = G.nodes[agent]['action']
            action_gains = {i: 0 for i in range(num_actions)}
            
            for change_action in range(num_actions):
                for neighbor in G.neighbors(agent):
                    # receive information from neighbors
                    neighbor_action = G.nodes[neighbor]['action']
                    # calculate the maximum decrease in cost if action is changed
                    cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                    change_action_cost = G[agent][neighbor]['cost_table'][change_action][neighbor_action]
                    action_gains[change_action] +=  cur_cost - change_action_cost

                #choose 
                # temperature = 5
                p = np.exp(action_gains[change_action] / temperature)
                probs.append(p)
            s = np.sum(probs)
            probs = probs / s
            chosen_action = np.random.choice([i for i in range(num_actions)], p=probs)
            G.nodes[agent]['action'] = chosen_action
        # Calculate the avg cost per iteration
        avg_cost = 0
        for u,v in G.edges():
            avg_cost += G[u][v]['cost_table'][G.nodes[u]['action']][G.nodes[v]['action']]
        avg_cost = avg_cost / len(G.edges())
        average_costs.append(avg_cost)

    return average_costs

def run_mgm_algorithm(G, num_iterations, num_actions):
    # Implement MGM algorithm here

# Then, it sends this information to all
# its neighbors. Upon receiving the values of its neighbors, it calculates the maximum gain (i.e., the
# maximum decrease in cost) if it changes its value and sends this information to all its neighbors.
# Upon receiving the gains of its neighbors, the agent changes its value if its gain is the largest among
# those of its neighbors. This process repeats until a termination condition is met. 

   
    average_costs = []
    for i in range(num_iterations):
        for agent in G.nodes():
            agent_action = G.nodes[agent]['action']
            max_agent_gain = 0
            optimal_action = agent_action
            # change_action = np.random.randint(0, 3)

            for change_action in range(num_actions):
                cur_gain = 0
                for neighbor in G.neighbors(agent):
                    # receive information from neighbors
                    neighbor_action = G.nodes[neighbor]['action']
                    # calculate the maximum decrease in cost if action is changed
                    cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                    change_action_cost = G[agent][neighbor]['cost_table'][change_action][neighbor_action]
                    cur_gain += cur_cost - change_action_cost
                if cur_gain > max_agent_gain:
                    max_agent_gain = cur_gain
                    optimal_action = change_action

            # get the maximum gain from neighbors
            max_neighbor_gain = 0
            for neighbor in G.neighbors(agent):
                neighbor_action = G.nodes[neighbor]['action']
                cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                change_action_cost = G[agent][neighbor]['cost_table'][optimal_action][neighbor_action]
                cur_gain = cur_cost - change_action_cost
                max_neighbor_gain = max(max_neighbor_gain, cur_gain)
            
            if max_neighbor_gain < max_agent_gain:
                # change action if agent gain is more than neighbors
                G.nodes[agent]['action'] = optimal_action

        # Calculate the avg cost
        avg_cost = 0
        for u,v in G.edges():
            avg_cost += G[u][v]['cost_table'][G.nodes[u]['action']][G.nodes[v]['action']]
        avg_cost = avg_cost / len(G.edges())
        average_costs.append(avg_cost)

    return average_costs

def run_experiment(num_agents, edge_density, graph_type, dsan_temperature, dsa_threshold, num_actions, niter, cost_range, experiment, path, file_name):

    num_instances = 10
    num_iterations = niter

    dsa_costs = []
    mgm_costs = []
    dsan_costs = []
    for _ in range(num_instances):
        if graph_type != 'erdos_renyi':
            G = create_random_dcop(num_agents, edge_density, graph_type)
        else:
            G = nx.erdos_renyi_graph(num_agents, p=edge_density)

        create_cost_tables(G, num_actions, cost_range)

        assign_initial_actions(G, num_actions)
        dsa_cost = run_dsa_algorithm(G, num_iterations, num_actions, dsa_threshold)

        assign_initial_actions(G, num_actions)
        dsan_cost = run_dsan_algorithm(G, num_iterations, num_actions, dsan_temperature)

        assign_initial_actions(G, num_actions)
        mgm_cost = run_mgm_algorithm(G, num_iterations, num_actions)

        dsa_costs.append(dsa_cost)
        mgm_costs.append(mgm_cost)
        dsan_costs.append(dsan_cost)
    
    
    # Average performance for each iteration across 10 instances, 
    avg_mgm_costs = []
    for i in range(num_iterations):
        total_cost = 0
        for j in range(num_instances):
            total_cost += mgm_costs[j][i]
        avg_cost = total_cost / num_instances
        avg_mgm_costs.append(avg_cost)

    avg_dsa_costs = []
    for i in range(num_iterations):
        total_cost = 0
        for j in range(num_instances):
            total_cost += dsa_costs[j][i]
        avg_cost = total_cost / num_instances
        avg_dsa_costs.append(avg_cost)

    avg_dsan_costs = []
    for i in range(num_iterations):
        total_cost = 0
        for j in range(num_instances):
            total_cost += dsan_costs[j][i]
        avg_cost = total_cost / num_instances
        avg_dsan_costs.append(avg_cost)

    # local minimas
    def local_minimas(costs):
        minimas = [costs[0]]
        for i in range(1, len(costs)):
            minimas.append(min(minimas[-1], costs[i]))
        return minimas
    
    min_dsa_costs = local_minimas(avg_dsa_costs)
    min_mgm_costs = local_minimas(avg_mgm_costs)
    min_dsan_costs = local_minimas(avg_dsan_costs)

    # avg_dsa_cost = np.mean(dsa_costs, axis=0)
    # avg_mgm_cost = np.mean(mgm_costs, axis=0)

    plt.plot(avg_dsa_costs, label='DSA')
    plt.plot(avg_mgm_costs, label='MGM')
    plt.plot(avg_dsan_costs, label = 'DSAN')
    # plt.plot(min_dsa_costs, label='DSA Minimas')
    # plt.plot(min_mgm_costs, label='MGM Minimas')
    # plt.plot(min_dsan_costs, label = 'DSAN Minimas')

    plt.xlabel('Iterations')
    plt.ylabel('Average Cost')
    plt.title(f'Algorithm Performance Comparison - {experiment} Experiment')
    # put to the right of the graph the parameters of the experiment in small font in a column
    plt.text(0.65, 0.865, f'Agents: {num_agents}\nEdge Density: {edge_density}\nDSAN Temperature: {dsan_temperature}\nDSA Threshold: {dsa_threshold}\nNum Actions: {num_actions}\nNum Iterations: {niter}\nCost Range: {cost_range}\n Graph: {graph_type}', fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.legend()
    # download the plot in /DCOP_GRAPHS folder
    plt.savefig(f'{path}/{file_name}.png')
    # clear the plot
    plt.clf()

def main():
    num_agents_list = [10, 25, 50, 75, 90, 100]
    edge_density_list = [0.1, 0.25, 0.5, 0.75, 0.9, .99]
    graph_types = ['complete','erdos_renyi', 'barabasi_albert',]
    dsan_temperatures = [0.75, 1.0, 2.0, 5.0, 10.0]
    dsa_thresholds = [0.1, 0.25, 0.5, .75, 0.9, 0.99]
    num_actions_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_iterations = [50, 100, 200, 500, 1000]
    cost_ranges = [4, 11, 51, 101, 501]

    for graph_type in graph_types:
        run_experiment(num_agents=100, edge_density=0.1, graph_type=graph_type, dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'graph_type', path = './DCOP_GRAPHS/By_graph_type',  file_name = graph_type)

    for num_agents in num_agents_list:
        run_experiment(num_agents=num_agents, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'num_agents', path = './DCOP_GRAPHS/By_num_agents',  file_name = f'{num_agents}_agents')
    
    for edge_density in edge_density_list:
        run_experiment(num_agents=100, edge_density=edge_density, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'edge_density', path = './DCOP_GRAPHS/By_edge_density',  file_name = f'{edge_density}_edge_density')
    
    for dsan_temperature in dsan_temperatures:
        run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=dsan_temperature, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'dsan_temperature', path = './DCOP_GRAPHS/By_dsan_temp',  file_name = f'{dsan_temperature}_dsan_temp')
    
    for dsa_threshold in dsa_thresholds:
        run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=dsa_threshold, num_actions=3, niter=200, cost_range=101, experiment = 'dsa_threshold', path = './DCOP_GRAPHS/By_dsa_threshold',  file_name = f'{dsa_threshold}_dsa_threshold')
    
    for num_actions in num_actions_list:
        run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=num_actions, niter=200, cost_range=101, experiment = 'num_actions', path = './DCOP_GRAPHS/By_num_actions',  file_name = f'{num_actions}_actions')
    
    for niter in num_iterations:
        run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=niter, cost_range=101, experiment = 'num_iterations', path = './DCOP_GRAPHS/By_num_iterations',  file_name = f'{niter}_iterations')
    
    for cost_range in cost_ranges:
          run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=cost_range, experiment = 'cost_range', path = './DCOP_GRAPHS/By_cost_range',  file_name = f'1-{cost_range}_cost_range')
    
    # for num_agents in num_agents_list:
    #     for edge_density in edge_density_list:
    #         for graph_type in graph_types:
    #             for dsan_temperature in dsan_temperatures:
    #                 for dsa_threshold in dsa_thresholds:
    #                     for num_actions in num_actions_list:
    #                         for niter in num_iterations:
    #                             for cost_range in cost_ranges:
    #                                 try:
    #                                     run_experiment(num_agents=num_agents, edge_density=edge_density, graph_type=graph_type, dsan_temperature=dsan_temperature, dsa_threshold=dsa_threshold, num_actions=num_actions, niter=niter, cost_range=cost_range, experiment = 'all graphs', path = './DCOP_GRAPHS/All_graphs',  file_name = f'{num_agents}_{edge_density}_{graph_type}_{dsan_temperature}_{dsa_threshold}_{num_actions}_{niter}_{cost_range}')
    #                                 except:
    #                                     with open('errors.txt', 'a') as f:
    #                                         f.write(f'Error in {num_agents}_{edge_density}_{graph_type}_{dsan_temperature}_{dsa_threshold}_{num_actions}_{niter}_{cost_range}\n')
    #                                     continue
if __name__ == "__main__":
    main()

### Notes:
# - The `run_dsa_algorithm` and `run_mgm_algorithm` functions need to be implemented based on the specific algorithms' details.
# - The cost assignment and edge selection methods should be adjusted as needed for the specific requirements of your benchmark and algorithms.



# NEXT
# Number of messages(x) vs. cost (y)
# Varying number of edge densities
# Varying number of nodes
# Varying type of graph structures (erdos_reyni, barabasi_albert, etc...)
# Make latex report of the results of the different algorithms, paragraph
# Play around with temperature param in DSAN (Best temperature for each graph structure randomizing edge density, nodes, etc...)
# play around with "threshold" value in DSA
# play around with num of actions (currently 3)
# Write a script to automate experiments above (do mock runs to see if it works)