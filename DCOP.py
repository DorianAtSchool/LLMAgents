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

def create_random_dcop(num_agents, edge_density):
    G = nx.complete_graph(num_agents)
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

def create_cost_tables(G):
    # Each edge has a cost table based on the possible actions of the agents (both agents have actions between 0-2, costing between 1- 10)
    for u, v in G.edges():
        G[u][v]['cost_table'] = np.random.randint(1, 11, size=(3, 3))

def assign_initial_actions(G):
    # Randomly assign initial actions to agents
    for agent in G.nodes():
        G.nodes[agent]['action'] = np.random.randint(0, 3)

def run_dsa_algorithm(G, num_iterations):
    # Implement DSA algorithm here
    average_costs = []
    for i in range(num_iterations):
        for agent in G.nodes():
            agent_action = G.nodes[agent]['action']
            # change_action = np.random.randint(0, 3)
            action_gains = {0: 0, 1: 0, 2: 0}
            for change_action in range(3):
                cur_gain = 0
                for neighbor in G.neighbors(agent):
                    # receive information from neighbors
                    neighbor_action = G.nodes[neighbor]['action']
                    # calculate the maximum gain if action is changed
                    cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                    change_action_cost = G[agent][neighbor]['cost_table'][change_action][neighbor_action]
                    action_gains[change_action] += change_action_cost - cur_cost
            action_gains = [key for key in action_gains.keys() if action_gains[key] > 0]
            #choose random index of action
            if len(action_gains) > 0:
                optimal_action = np.random.choice(action_gains)
            else:
                optimal_action = agent_action
            # get the maximum gain from neighbors
            # max_neighbor_gain = 0
            # for neighbor in G.neighbors(agent):
            #     neighbor_action = G.nodes[neighbor]['action']
            #     cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
            #     change_action_cost = G[agent][neighbor]['cost_table'][optimal_action][neighbor_action]
            #     cur_gain = change_action_cost - cur_cost
            #     max_neighbor_gain = max(max_neighbor_gain, cur_gain)
            
            # if max_neighbor_gain < max_agent_gain:
            #     # change action if agent gain is more than neighbors
            #     G.nodes[agent]['action'] = optimal_action
            G.nodes[agent]['action'] = optimal_action
            # Calculate the avg cost
            avg_cost = 0
            for u,v in G.edges():
                avg_cost += G[u][v]['cost_table'][G.nodes[u]['action']][G.nodes[v]['action']]
            avg_cost = avg_cost / len(G.edges())
            average_costs.append(avg_cost)

    return average_costs
def run_mgm_algorithm(G, num_iterations):
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

            for change_action in range(3):
                cur_gain = 0
                for neighbor in G.neighbors(agent):
                    # receive information from neighbors
                    neighbor_action = G.nodes[neighbor]['action']
                    # calculate the maximum gain if action is changed
                    cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                    change_action_cost = G[agent][neighbor]['cost_table'][change_action][neighbor_action]
                    cur_gain += change_action_cost - cur_cost
                if cur_gain > max_agent_gain:
                    max_agent_gain = cur_gain
                    optimal_action = change_action

            # get the maximum gain from neighbors
            max_neighbor_gain = 0
            for neighbor in G.neighbors(agent):
                neighbor_action = G.nodes[neighbor]['action']
                cur_cost = G[agent][neighbor]['cost_table'][agent_action][neighbor_action]
                change_action_cost = G[agent][neighbor]['cost_table'][optimal_action][neighbor_action]
                cur_gain = change_action_cost - cur_cost
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

def main():
    num_agents = 10
    edge_density = 0.2
    num_instances = 10
    num_iterations = 1000
    dsa_costs = []
    mgm_costs = []
    for _ in range(num_instances):
        G = create_random_dcop(num_agents, edge_density)
        create_cost_tables(G)
        assign_initial_actions(G)
        print(G.edges(data=True))
        dsa_cost = run_dsa_algorithm(G, num_iterations)
        assign_initial_actions(G)
        mgm_cost = run_mgm_algorithm(G, num_iterations)
        dsa_costs.append(dsa_cost)
        mgm_costs.append(mgm_cost)
    
    
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

    avg_dsa_cost = np.mean(dsa_costs, axis=0)
    avg_mgm_cost = np.mean(mgm_costs, axis=0)

    plt.plot(avg_dsa_costs, label='DSA')
    plt.plot(avg_mgm_costs, label='MGM')
    plt.xlabel('Iterations')
    plt.ylabel('Average Cost')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()

### Notes:
# - The `run_dsa_algorithm` and `run_mgm_algorithm` functions need to be implemented based on the specific algorithms' details.
# - The cost assignment and edge selection methods should be adjusted as needed for the specific requirements of your benchmark and algorithms.



