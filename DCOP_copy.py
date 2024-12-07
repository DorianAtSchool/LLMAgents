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
from tqdm import tqdm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

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
                # action_gains = np.array(action_gains)
                
                p = np.exp((action_gains[change_action]) / temperature)
                probs.append(p)
            
    
            s = np.sum(probs)
            probs = probs / s

            # try:
            chosen_action = np.random.choice([i for i in range(num_actions)], p= probs)
            # except:
            #     print((action_gains[change_action]) )
            #     print(np.exp(action_gains[change_action] / temperature))
            #     print(probs)
            #     print(s)
            #     best_action = np.argmax(probs)
            #     chosen_action = best_action

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

    num_instances = 3
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


def run_dsan_experiment(num_agents, edge_density, graph_type, dsan_temperature, num_actions, niter, cost_range, experiment, path, file_name):
    

    # POTENTIAL ISSUE: When creating the dataset I used 10 instances whereas to run the model predictions I only used 1 instance. This could be a source of error.
    num_instances = 3
    num_iterations = niter

    dsan_costs = []
    for _ in range(num_instances):
        if graph_type != 'erdos_renyi':
            G = create_random_dcop(num_agents, edge_density, graph_type)
        else:
            G = nx.erdos_renyi_graph(num_agents, p=edge_density)

        create_cost_tables(G, num_actions, cost_range)

        assign_initial_actions(G, num_actions)
        
        
        dsan_cost = run_dsan_algorithm(G, num_iterations, num_actions, dsan_temperature)
        dsan_costs.append(dsan_cost)
        

    # Average performance for each iteration across 10 instances
    avg_dsan_costs = []
    for i in range(num_iterations):
        total_cost = 0
        for j in range(len(dsan_costs)):
            total_cost += dsan_costs[j][i]
        avg_cost = total_cost / len(dsan_costs)
        avg_dsan_costs.append(avg_cost)


    min_cost = min(avg_dsan_costs)
    
    return min_cost

def test_dsan():
    
    num_instances = 3
    num_iterations = 500

    dsan_costs = []
    for _ in range(num_instances):
        G = nx.erdos_renyi_graph(75, p=.9)
        
        create_cost_tables(G, 9, 101)

        assign_initial_actions(G, 9)
        
        dsan_cost = run_dsan_algorithm(G, num_iterations, 9, 1)
        dsan_costs.append(dsan_cost)
        

    
    
    # Average performance for each iteration across 10 instances
    avg_dsan_costs = []
    for i in range(num_iterations):
        total_cost = 0
        for j in range(num_instances):
            total_cost += dsan_costs[j][i]
        avg_cost = total_cost / num_instances
        avg_dsan_costs.append(avg_cost)


    min_cost = min(avg_dsan_costs)
    
    return min_cost

def run_dsa_experiment(num_agents, edge_density, graph_type, dsa_threshold, num_actions, niter, cost_range, experiment, path, file_name):
        
        num_instances = 3
        num_iterations = niter
    
        dsa_costs = []
        for _ in range(num_instances):
            if graph_type != 'erdos_renyi':
                G = create_random_dcop(num_agents, edge_density, graph_type)
            else:
                G = nx.erdos_renyi_graph(num_agents, p=edge_density)
    
            create_cost_tables(G, num_actions, cost_range)
    
            assign_initial_actions(G, num_actions)
            dsa_cost = run_dsa_algorithm(G, num_iterations, num_actions, dsa_threshold)
            dsa_costs.append(dsa_cost)
        
        
        # Average performance for each iteration across 10 instances
        avg_dsa_costs = []
        for i in range(num_iterations):
            total_cost = 0
            for j in range(num_instances):
                total_cost += dsa_costs[j][i]
            avg_cost = total_cost / num_instances
            avg_dsa_costs.append(avg_cost)
    
    
        min_cost = min(avg_dsa_costs)
        
        return min_cost

# create a dataset for performance prediction
def create_dataset_dsan():
    # DSA - num agents, cost range, edge density, graph type, number of action choices, DSAN's temp, num iterations (x features) vs. best cost (y)
    # ^^ pick random from list for each feature (so for num agents: pick random number from 1-100) -> run DSA -> get best Y value across all itrations
    # repeat ~ 1000 times or however many
    num_agents_list = [10, 25, 50, 75, 90, 100]
    edge_density_list = [0.1, 0.25, 0.5, 0.75, 0.9, .99]
    graph_types = ['complete','erdos_renyi', 'barabasi_albert',]
    dsan_temperatures = [0.75, 1.0, 2.0, 5.0, 10.0]
    num_actions_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_iterations = [50, 100, 200, 500, 1000]
    cost_ranges = [4, 11, 51, 101, 501]

    # make a df for x = features, y = best cost
    with open('./DCOP_GRAPHS/DSAN/dsan_dataset_2.csv', 'w') as f:
        f.write('num_agents,edge_density,graph_type,dsan_temperature,num_actions,num_iterations,cost_range,y\n')

    # Initialize the progress bar with the total number of iterations
    total_iterations = 1000
    pbar = tqdm(total=total_iterations)

    i = 0
    while i < 1000:
        n_agent = np.random.choice(num_agents_list)
        e_density = np.random.choice(edge_density_list)
        g_type = np.random.choice(graph_types)
        d_temp = np.random.choice(dsan_temperatures)
        n_actions = np.random.choice(num_actions_list)
        n_iter = np.random.choice(num_iterations)
        c_range = np.random.choice(cost_ranges)
        try:
            y = run_dsan_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsan_temperature=d_temp, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsan_dataset', path = './DCOP_GRAPHS/DSAN',  file_name = f'{n_agent}_{e_density}_{g_type}_{d_temp}_{n_actions}_{n_iter}_{c_range}')
            with open('./DCOP_GRAPHS/DSAN/dsan_dataset_2.csv', 'a') as f:
                f.write(f'{n_agent},{e_density},{g_type},{d_temp},{n_actions},{n_iter},{c_range},{y}\n')
            i += 1
            pbar.update(1)  
        except:
            print('error', i)
            continue
        

# create a dataset for performance prediction
def create_dataset_dsa():
    # DSA - num agents, cost range, edge density, graph type, number of action choices, DSAN's temp, num iterations (x features) vs. best cost (y)
    # ^^ pick random from list for each feature (so for num agents: pick random number from 1-100) -> run DSA -> get best Y value across all itrations
    # repeat ~ 1000 times or however many
    num_agents_list = [10, 25, 50, 75, 90, 100]
    edge_density_list = [0.1, 0.25, 0.5, 0.75, 0.9, .99]
    graph_types = ['complete','erdos_renyi', 'barabasi_albert',]
    dsa_thresholds = [0.1, 0.25, 0.5, .75, 0.9, 0.99]    
    num_actions_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_iterations = [50, 100, 200, 500, 1000]
    cost_ranges = [4, 11, 51, 101, 501]

    # make a df for x = features, y = best cost
    with open('./DCOP_GRAPHS/DSA/dsa_dataset_2.csv', 'w') as f:
        f.write('num_agents,edge_density,graph_type,dsa_threshold,num_actions,num_iterations,cost_range,y\n')

    # Initialize the progress bar with the total number of iterations
    total_iterations = 1000
    pbar = tqdm(total=total_iterations)

    i = 0
    while i < 1000:
        n_agent = np.random.choice(num_agents_list)
        e_density = np.random.choice(edge_density_list)
        g_type = np.random.choice(graph_types)
        d_thresh = np.random.choice(dsa_thresholds)
        n_actions = np.random.choice(num_actions_list)
        n_iter = np.random.choice(num_iterations)
        c_range = np.random.choice(cost_ranges)

        try:
            y = run_dsa_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsa_threshold=d_thresh, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsan_dataset', path = './DCOP_GRAPHS/DSAN',  file_name = f'{n_agent}_{e_density}_{g_type}_{d_thresh}_{n_actions}_{n_iter}_{c_range}')
            with open('./DCOP_GRAPHS/DSA/dsa_dataset_2.csv', 'a') as f:
                f.write(f'{n_agent},{e_density},{g_type},{d_thresh},{n_actions},{n_iter},{c_range},{y}\n')
            i += 1
            pbar.update(1)  
        except:
            print('error', i, 'params:', n_agent, e_density, g_type, d_thresh, n_actions, n_iter, c_range)
            continue
        


def train_decision_tree(dataset):
    # Prepare the dataset for training 
    # map graph types to numerical values: complete=0, erdos_renyi=1, barabasi_albert=2
    dataset['graph_type'] = dataset['graph_type'].map({'complete': 0, 'erdos_renyi': 1, 'barabasi_albert': 2})
    # convert y to int for classification
    dataset['y'] = dataset['y']

    # Get the features and target variable
    Y = dataset['y']
    X = dataset.drop('y', axis=1)
    

    # Split to train and test  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Train the decision tree classifier
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    # accuracy = accuracy_score(y_train, y_pred)
    accuracy = mean_squared_error(y_train, y_pred)
    print(f'Training Accuracy: {accuracy}')

    # Test the model
    y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    accuracy = mean_squared_error(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')

    # Save the model
    joblib.dump(clf, 'dsan_model.pkl')
    return clf

def main():
    num_agents_list = [10, 25, 50, 75, 90, 100]
    edge_density_list = [0.1, 0.25, 0.5, 0.75, 0.9, .99]
    graph_types = ['complete','erdos_renyi', 'barabasi_albert',]
    dsan_temperatures = [0.75, 1.0, 2.0, 5.0, 10.0]
    dsa_thresholds = [0.1, 0.25, 0.5, .75, 0.9, 0.99]
    num_actions_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_iterations = [50, 100, 200, 500, 1000]
    cost_ranges = [4, 11, 51, 101, 501]


    # INITIAL EXPERIMENTS

    # for graph_type in graph_types:
    #     run_experiment(num_agents=100, edge_density=0.1, graph_type=graph_type, dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'graph_type', path = './DCOP_GRAPHS/By_graph_type',  file_name = graph_type)

    # for num_agents in num_agents_list:
    #     run_experiment(num_agents=num_agents, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'num_agents', path = './DCOP_GRAPHS/By_num_agents',  file_name = f'{num_agents}_agents')
    
    # for edge_density in edge_density_list:
    #     run_experiment(num_agents=100, edge_density=edge_density, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'edge_density', path = './DCOP_GRAPHS/By_edge_density',  file_name = f'{edge_density}_edge_density')
    
    # for dsan_temperature in dsan_temperatures:
    #     run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=dsan_temperature, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=101, experiment = 'dsan_temperature', path = './DCOP_GRAPHS/By_dsan_temp',  file_name = f'{dsan_temperature}_dsan_temp')
    
    # for dsa_threshold in dsa_thresholds:
    #     run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=dsa_threshold, num_actions=3, niter=200, cost_range=101, experiment = 'dsa_threshold', path = './DCOP_GRAPHS/By_dsa_threshold',  file_name = f'{dsa_threshold}_dsa_threshold')
    
    # for num_actions in num_actions_list:
    #     run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=num_actions, niter=200, cost_range=101, experiment = 'num_actions', path = './DCOP_GRAPHS/By_num_actions',  file_name = f'{num_actions}_actions')
    
    # for niter in num_iterations:
    #     run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=5.0, dsa_threshold=0.95, num_actions=3, niter=niter, cost_range=101, experiment = 'num_iterations', path = './DCOP_GRAPHS/By_num_iterations',  file_name = f'{niter}_iterations')
    
    # for cost_range in tqdm(cost_ranges):
    #       run_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=1.0, dsa_threshold=0.95, num_actions=3, niter=200, cost_range=cost_range, experiment = 'cost_range', path = './DCOP_GRAPHS/By_cost_range',  file_name = f'1-{cost_range}_cost_range')
    
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

    ############################################################################################################

    # PLOT DSAN TEMPERATURE VS. MIN COST and DSA THRESHOLD VS. MIN COST

    # points = {}
    # for t in dsan_temperatures:
    #     min_cost = run_dsan_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsan_temperature=t, num_actions=3, niter=200, cost_range=101, experiment = 'dsan_temperature', path = './DCOP_GRAPHS/DSAN',  file_name = f'{t}_dsan_temp')
    #     points[t] = min_cost
    
    # plt.plot(points.keys(), points.values())
    # plt.xlabel('Temperature')
    # plt.ylabel('Min Cost')
    # plt.title('DSAN Temperature vs. Min Cost')
    # plt.savefig('./DCOP_GRAPHS/DSAN/DSAN_Temperature_vs_Min_Cost.png')
    # plt.show()
    # plt.clf()

    # points = {}
    # for t in dsa_thresholds:
    #     min_cost = run_dsa_experiment(num_agents=100, edge_density=0.1, graph_type='complete', dsa_threshold=t, num_actions=3, niter=200, cost_range=101, experiment = 'dsa_threshold', path = './DCOP_GRAPHS/DSA',  file_name = f'{t}_dsa_threshold')
    #     points[t] = min_cost
    
    # plt.plot(points.keys(), points.values())
    # plt.xlabel('Threshold')
    # plt.ylabel('Min Cost')
    # plt.title('DSA Threshold vs. Min Cost')
    # plt.savefig('./DCOP_GRAPHS/DSA/DSA_Threshold_vs_Min_Cost.png')
    # plt.show()
    # plt.clf()

    ############################################################################################################

    # PERFORMANCE PREDICTION

    #   DSAN PERFORMANCE PREDICTION
    # create_dataset_dsan()
    # model = train_decision_tree(pd.read_csv('./DCOP_GRAPHS/DSAN/dsan_dataset.csv'))
    
    # # Evaluate Model: keep other vairbales constant and change temp in increments of say .1 from .1 to 10 to make prediction. 
    # # pick best temp value based on those predictions
    # # run dsan using that paramter for that problem
    # # create 1000 test DCOP problems, predict temp paramter for each, run DSAN for that problem, get best cost, average best costs across 1000 problems for a single value

    # t_vals = np.arange(0.1, 10.1, 0.1)
    # t_costs = { t: [] for t in t_vals}
    # best_costs = []
    # pbar = tqdm(total=1000)
    # i = 0
    # while i < 1000:
    #     # generate a random problem
    #     n_agent = np.random.choice(num_agents_list)
    #     e_density = np.random.choice(edge_density_list)
    #     g_type = np.random.choice(graph_types)
    #     n_actions = np.random.choice(num_actions_list)
    #     n_iter = np.random.choice(num_iterations)
    #     c_range = np.random.choice(cost_ranges)

    #     #preprocess g_type to 0,1,2
    #     g_type_num = {'complete': 0, 'erdos_renyi': 1, 'barabasi_albert': 2}[g_type]

    #     # make predictions
    #     predictions = {}
    #     for t in t_vals:
    #         df = pd.DataFrame({'num_agents': [n_agent], 'edge_density': [e_density], 'graph_type': [g_type_num], 'dsan_temperature': [t], 'num_actions': [n_actions], 'num_iterations': [n_iter], 'cost_range': [c_range]})
    #         # make a prediction
    #         y_pred = model.predict(df)
    #         predictions[t] = y_pred[0]

    #     best_temp = min(predictions, key=predictions.get)
    #     t_costs[best_temp].append(predictions[best_temp])

    #     try:
    #         min_cost = run_dsan_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsan_temperature=best_temp, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsan_dataset', path = './DCOP_GRAPHS/DSAN',  file_name = f'{n_agent}_{e_density}_{g_type}_{best_temp}_{n_actions}_{n_iter}_{c_range}')
    #         best_costs.append(min_cost)
    #         i += 1
    #         pbar.update(1)
    #     except:
    #         print('error', i)
    #         continue
        

    # best_cost = np.mean(best_costs)
    # print(best_cost)

    # t_costs = {k: sum(v)/len(v) for k,v in t_costs.items() if len(v) > 0}
    # plt.bar(t_costs.keys(), t_costs.values())
    # plt.xlabel('Predicted Temperature')
    # plt.ylabel('Avg Best Cost')
    # plt.title('Predicted DSAN Temperature vs. Avg Best Cost')
    # plt.savefig('./DCOP_GRAPHS/DSAN/Predicted_DSAN_Temperature_vs_Avg_Best_Cost.png')
    # plt.clf()

    # # create 1000 test DCOP problems, choose a static temp paramter for all problems, run DSAN for that problem, get best cost, average best costs across 1000 problems for a single value
    # # repeat line above 5 times with different  static temps
    # pbar = tqdm(total=1000)
    # pbar_t = tqdm(total=5)

    # dsan_temperatures = [0.75, 1.0, 2.0, 5.0, 10.0]
    # t_costs = {}
    # for t in dsan_temperatures:
    #     best_costs = []
    #     i = 0
    #     while i < 1000:
    #         # generate a random problem
    #         n_agent = np.random.choice(num_agents_list)
    #         e_density = np.random.choice(edge_density_list)
    #         g_type = np.random.choice(graph_types)
    #         n_actions = np.random.choice(num_actions_list)
    #         n_iter = np.random.choice(num_iterations)
    #         c_range = np.random.choice(cost_ranges)

    #         try:
    #             min_cost = run_dsan_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsan_temperature=t, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsan_dataset', path = './DCOP_GRAPHS/DSAN',  file_name = f'{n_agent}_{e_density}_{g_type}_{t}_{n_actions}_{n_iter}_{c_range}')
    #             best_costs.append(min_cost)
    #             i += 1
    #             pbar.update(1)
    #         except:
    #             print('error', i)
    #             continue
            
    #     best_cost = np.mean(best_costs)
    #     t_costs[t] = best_cost
    #     pbar_t.update(1)
        
    # # make bar graphs of threshold vs. final value
    # plt.bar(t_costs.keys(), t_costs.values())
    # plt.xlabel('Static Temperature')
    # plt.ylabel('Avg Best Cost')
    # plt.title('Static DSAN Temperature vs. Avg Best Cost')
    # plt.savefig('./DCOP_GRAPHS/DSAN/Static_DSAN_Temperature_vs_Avg_Best_Cost.png')
    # plt.show()
    # plt.clf()

    #  DSA PERFORMANCE PREDICTION

    # create_dataset_dsa()
    # create_dataset_dsan()

    model = train_decision_tree(pd.read_csv('./DCOP_GRAPHS/DSA/dsa_dataset_2.csv'))

    # use the model to check if it predicts correctly for live data

    t_vals = [0.1, 0.25, 0.5, .75, 0.9, 0.99]
    t_costs = { t: [] for t in t_vals}
    best_costs = []
    pbar = tqdm(total=1000)
    i = 0
    while i < 1:
        # generate a random problem
        n_agent = np.random.choice(num_agents_list)
        e_density = np.random.choice(edge_density_list)
        g_type = np.random.choice(graph_types)
        n_actions = np.random.choice(num_actions_list)
        n_iter = np.random.choice(num_iterations)
        c_range = np.random.choice(cost_ranges)

        #preprocess g_type to 0,1,2
        g_type_num = {'complete': 0, 'erdos_renyi': 1, 'barabasi_albert': 2}[g_type]

        # make predictions and experiments
        actual_costs = {}
        predictions = {}
        for t in t_vals:
            df = pd.DataFrame({'num_agents': [n_agent], 'edge_density': [e_density], 'graph_type': [g_type_num], 'dsa_threshold': [t], 'num_actions': [n_actions], 'num_iterations': [n_iter], 'cost_range': [c_range]})
            # make a prediction
            y_pred = model.predict(df)
            predictions[t] = y_pred[0]
            # run the experiment
            actual_cost = run_dsa_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsa_threshold=t, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsa_dataset', path = './DCOP_GRAPHS/DSA',  file_name = f'{n_agent}_{e_density}_{g_type}_{t}_{n_actions}_{n_iter}_{c_range}')
            actual_costs[t] = actual_cost

        print("predictions: ", predictions)
        print("experiment costs: ", actual_costs)

        pred_best_threshold = min(predictions, key=predictions.get)
        print( "best predicted t: ", pred_best_threshold)
        actual_best_threshold = min(actual_costs, key=actual_costs.get)
        print("best experioment t", actual_best_threshold)

        plt.plot(predictions.keys(), predictions.values(), label='Predicted Costs')
        plt.plot(actual_costs.keys(), actual_costs.values(), label='Actual Costs')
        plt.xlabel('Threshold')
        plt.ylabel('Cost')
        plt.title('Predicted vs. Actual Costs')
        plt.show()
        plt.clf()
        i += 1
        pbar.update(1)



    # Evaluate Model: keep other varibales constant and change threshold in increments of say .1 from .1 to 10 to make prediction.
    # pick best threshold value based on those predictions
    # run dsa using that paramter for that problem

    # t_vals = np.arange(0.01, 1.1, .01) # This has to be between 0-1 since dataset was created with that range
    t_vals = [0.1, 0.25, 0.5, .75, 0.9, 0.99]
    t_costs = { t: [] for t in t_vals}
    best_costs = []
    pbar = tqdm(total=1000)
    i = 0
    while i < 50:
        # generate a random problem
        n_agent = np.random.choice(num_agents_list)
        e_density = np.random.choice(edge_density_list)
        g_type = np.random.choice(graph_types)
        n_actions = np.random.choice(num_actions_list)
        n_iter = np.random.choice(num_iterations)
        c_range = np.random.choice(cost_ranges)

        #preprocess g_type to 0,1,2
        g_type_num = {'complete': 0, 'erdos_renyi': 1, 'barabasi_albert': 2}[g_type]

        # make predictions
        predictions = {}
        for t in t_vals:
            df = pd.DataFrame({'num_agents': [n_agent], 'edge_density': [e_density], 'graph_type': [g_type_num], 'dsa_threshold': [t], 'num_actions': [n_actions], 'num_iterations': [n_iter], 'cost_range': [c_range]})
            # make a prediction
            y_pred = model.predict(df)
            predictions[t] = y_pred[0]
        print(predictions)
        best_threshold = min(predictions, key=predictions.get)
        print(best_threshold)
        t_costs[best_threshold].append(predictions[best_threshold])

        # try:
        min_cost = run_dsa_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsa_threshold=best_threshold, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsa_dataset', path = './DCOP_GRAPHS/DSA',  file_name = f'{n_agent}_{e_density}_{g_type}_{best_threshold}_{n_actions}_{n_iter}_{c_range}')
        best_costs.append(min_cost)
        i += 1
        #     pbar.update(1)
        # except:
        #     print('error', i)
        #     continue
        

    best_cost = np.mean(best_costs)
    print(best_cost)

    t_costs = {k: sum(v)/len(v) for k,v in t_costs.items() if len(v) > 0}
    print(t_costs)
    plt.bar(t_costs.keys(), t_costs.values())
    plt.xlabel('Predicted Threshold')
    plt.ylabel('Avg Best Cost')
    plt.title('Predicted DSA Threshold vs. Avg Best Cost')
    plt.savefig('./DCOP_GRAPHS/DSA/Predicted_DSA_Threshold_vs_Avg_Best_Cost_2.png')
    plt.show()
    plt.clf()

    # # create 1000 test DCOP problems, choose a static temp paramter for all problems, run DSAN for that problem, get best cost, average best costs across 1000 problems for a single value
    # # repeat line above 5 times with different  static temps
    # pbar = tqdm(total=1000)
    # pbar_t = tqdm(total=6)

    # dsa_thresholds = [0.1, 0.25, 0.5, .75, 0.9, 0.99]
    # t_costs = {}
    # for t in dsa_thresholds:
    #     best_costs = []
    #     i = 0
    #     while i < 1000:
    #         # generate a random problem
    #         n_agent = np.random.choice(num_agents_list)
    #         e_density = np.random.choice(edge_density_list)
    #         g_type = np.random.choice(graph_types)
    #         n_actions = np.random.choice(num_actions_list)
    #         n_iter = np.random.choice(num_iterations)
    #         c_range = np.random.choice(cost_ranges)

    #         #preprocess g_type to 0,1,2
    #         # g_type_num = {'complete': 0, 'erdos_renyi': 1, 'barabasi_albert': 2}[g_type]

    #         try:
    #             min_cost = run_dsa_experiment(num_agents=n_agent, edge_density=e_density, graph_type=g_type, dsa_threshold=t, num_actions=n_actions, niter=n_iter, cost_range=c_range, experiment = 'dsa_dataset', path = './DCOP_GRAPHS/DSA',  file_name = f'{n_agent}_{e_density}_{g_type}_{t}_{n_actions}_{n_iter}_{c_range}')
    #             best_costs.append(min_cost)
    #             i += 1
    #             pbar.update(1)
    #         except:
    #             print('error', i)
    #             continue
            
    #     best_cost = np.mean(best_costs)
    #     t_costs[t] = best_cost
    #     pbar_t.update(1)
        
    # # make bar graphs of threshold vs. final value
    # plt.bar(t_costs.keys(), t_costs.values())
    # plt.xlabel('Static Threshold')
    # plt.ylabel('Avg Best Cost')
    # plt.title('Static DSA Threshold vs. Avg Best Cost')
    # plt.savefig('./DCOP_GRAPHS/DSA/Static_DSA_Threshold_vs_Avg_Best_Cost.png')
    # plt.show()
    # plt.clf()


    # test_dsan()
if __name__ == "__main__":
    main()

### Notes:
# - The `run_dsa_algorithm` and `run_mgm_algorithm` functions need to be implemented based on the specific algorithms' details.
# - The cost assignment and edge selection methods should be adjusted as needed for the specific requirements of your benchmark and algorithms.



# COMPLETED
# Number of messages(x) vs. cost (y) -- NOT COMPLETED
# Varying number of edge densities
# Varying number of nodes
# Varying type of graph structures (erdos_reyni, barabasi_albert, etc...)
# Make latex report of the results of the different algorithms, paragraph
# Play around with temperature param in DSAN (Best temperature for each graph structure randomizing edge density, nodes, etc...)
# play around with "threshold" value in DSA
# play around with num of actions (currently 3)
# Write a script to automate experiments above (do mock runs to see if it works)


##################################################################

# TO DO
# DSAN: Temp (x) vs.  best cost (y) graph
# DSA: threshold vs. best cost
# enlarge graphs
# create a dataset for performance prediction
    # DSA - num agents, cost range, edge density, graph type, number of action choices, DSA's throshold paramter , num iterations (x features) vs. best cost (y)
    # ^^ pick random from list for each feature (so for num agents: pick random number from 1-100) -> run DSA -> get best Y value across all itrations
    # repeat ~ 1000 times or however many
    # train neural network / decision tree/ regression to predict future performance
    # sklearn have those
    # Evaluate mode: optimize DSA threshold paramter using this model
    # for example, keep other vairbales constant and change threshold in increments of say .1 from .1 to 10 to make prediction. 
    # pick best temp value based on those predicitons
    # run dsa using that paramter for that problem
    # create 1000 test DCOP problems, predict threshold paramter for each, run DSA for that problem, get best cost, average best costs across 1000 problems for a single value
    # create 1000 test DCOP problems, choose a static thrshold paramter for all problems, run DSA for that problem, get best cost, average best costs across 1000 problems for a single value
    # repeat line above 5 times with different  static thrsholds
    # make bar graphs of threshold vs. final value

# do same for DSAN

################

# Python multiprocesses to parrallalize experiments, Julia for faster computation
# Debug DSAN algorithm
# 3-5 instances of each experiment
# Static and prediction bars vs cost - graph
# get rid of negative vals
# LOOK AT LINES 629-630
# Start small with ~ 50 tests

# Potential breaking issue: .99 theoretically alyways has the best cost so model always predicts it  