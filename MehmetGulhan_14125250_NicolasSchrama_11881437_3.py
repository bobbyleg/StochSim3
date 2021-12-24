import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

############################################# Experimental setup

# Parameters values
mc_length = 100000         # length of the markov chain
n = 50                  # nunber of simulations
T0 = mc_length // 1000
alpha1 = 1 - 10 / mc_length          # 0.9999 when mc_length = 100000
alpha2 = 1 - 5 / mc_length          # 0.99995 when mc_length = 100000 
b = (1 - alpha1) * 3 / 10 ** (5 - np.log10(mc_length))
cooling_scheme_values = f'geometrical {alpha1}'     # cooling schemes: geometrical with alpha1, geometrical with alpha2, arithmetic-geometric, linear

# Looping over one parameter
loop_var = 'mc_length' #mc_length' #'cooling_scheme_values'
if loop_var == 'mc_length':
    mc_length = [1000, 5000, 10000, 25000, 50000, 100000, 150000]
elif loop_var == 'T0':
    T0 = [1, mc_length // 10000, mc_length // 1000, mc_length // 100] 
else:
    cooling_scheme_values = [f'geometrical {alpha1}', f'geometrical with {alpha2}', 'arithmetic-geometric', 'linear']

# Load in data
data_eil51 = pd.read_csv("TSP-Configurations/eil51.tsp.txt")
data = np.array(data_eil51.iloc[5:-1])

data_a280 = pd.read_csv("TSP-Configurations/a280.tsp.txt")
data2 = np.array(data_a280.iloc[5:-1])

############################################# Simulated annealing

# Organize data in dictionary
def getSeperateIntCoords(nodes):
    new_nodes=[]
    for string in nodes:
        new_nodes.append(string[0].split(" "))

    # this part is only necessary for a280 data
    y = []
    for list in new_nodes:
        z = [x for x in list if x != '']
        y.append(z)
    new_nodes = y

    return np.array(new_nodes).astype(int)

def getNodesDict(nodes):
    nodes_dict = {}
    for node in nodes:
        nodes_dict[node[0]] = (node[1], node[2])
    return nodes_dict

# Create a random initial route
def getRandomRoute(nodes):
    random_route = np.arange(1, nodes+1, 1)
    np.random.shuffle(random_route)
    return random_route

# Perform 2 opt change in route
def generate_i_and_k(route):
    random_ints = np.random.choice(np.arange(0, len(route)), replace=False, size=(2, 1))
    i = np.min(random_ints)
    k = np.max(random_ints)
    return i, k

def two_opt_swap(route):
    i,k = generate_i_and_k(route)
    new_route = np.zeros(len(route))
    new_route[:i] = route[:i]
    new_route[i:k] = route[i:k][::-1]
    new_route[k:] = route[k:]
    return new_route

# Compute distance between two cities
def getDistance(node1, node2):
    diffx = np.abs(node1[0] - node2[0])
    diffy = np.abs(node1[1] - node2[1])
    return np.sqrt(diffx**2 + diffy**2)

# Calculate length of route
def objective(route, nodes_dict):
    length = 0
    for index, city in enumerate(route[:-1]):
        city1 = route[index]
        city2 = route[index+1]
        length += getDistance(nodes_dict[city1], nodes_dict[city2])
    length += getDistance(nodes_dict[route[0]], nodes_dict[route[-1]])
    return length

# Change temperature of cooling scheme
def cooling_scheme(T0, alpha1, alpha2, b, state, scheme, mc_length, temperatures):
    # exponential multiplicative with alpha1 
    if scheme == f'geometrical {alpha1}':
        return T0 * alpha1 ** state
    # exponential multiplicative with alpha2 
    if scheme == f'geometrical with {alpha2}':
        return T0 * alpha2 ** state
    # arithmetic-geometric cooling schedule
    elif scheme == 'arithmetic-geometric':
        return alpha2 * temperatures[-1] + b
    # linear 
    else:
        return T0 * ((mc_length - state) / mc_length)

# Create Markov chain of 2 opt swaps
def markov_chain(mc_length, T0, alpha1, alpha2, b, scheme, nodes, nodes_dict):
    route = getRandomRoute(len(nodes))
    lengths = np.array([objective(route, nodes_dict)])    
    T = T0
    temperatures = np.array([T0])
    
    for state in range(mc_length):
        potential_route = two_opt_swap(route)
        length_of_new_route = objective(potential_route, nodes_dict)

        alpha = np.exp(-(length_of_new_route -lengths[-1]) / T)
        if alpha > 1:
            lengths = np.append(lengths, length_of_new_route)
            route = potential_route
        else:
            draw = np.random.random()
            if draw < alpha:
                lengths = np.append(lengths, length_of_new_route)
                route = potential_route
            else:
                lengths = np.append(lengths, lengths[-1])

        T = cooling_scheme(T0, alpha1, alpha2, b, state, scheme, mc_length, temperatures)
        temperatures = np.append(temperatures, T)
    return route, lengths

# Perform many simulations
def sim(data, n, mc_length, T0, alpha1, alpha2, b, scheme):
    nodes = getSeperateIntCoords(data)
    nodes_dict = getNodesDict(nodes)

    all_routes = np.ones((n, len(nodes)))
    length_dict = {}

    best_route = np.ones(len(nodes))
    distance_br = 20000
    
    for simulation in tqdm(range(n)):
        chain_route, route_lengths = markov_chain(mc_length, T0, alpha1, alpha2, b, scheme, nodes, nodes_dict)
        all_routes[simulation] = chain_route
        for index, length in enumerate(route_lengths):
            if simulation == 0:
                length_dict[index] = np.ones(n)
                length_dict[index][0] = length 
            else:
                length_dict[index][simulation] = length 
        
        if route_lengths[-1] < distance_br:
            distance_br = route_lengths[-1]
            best_route = chain_route

    return all_routes, length_dict, best_route, distance_br

############################################# Plotting

# Plot evolution of route length over markov chain states per method with confidence intervals
def plot_means(loop_dict, mc_length, n, T0, loop_array, loop_var):
    colors = ["Blue", 'Yellow', 'Green', 'Red']
    count2 = 0

    for length_dict in loop_dict.items():
        means_per_state = np.ones(mc_length + 1)

        count1 = 0
        for state in length_dict[1].items():
            average_length = np.mean(state[1])
            means_per_state[count1] = average_length

            count1 += 1

        x = np.linspace(0, mc_length + 1, mc_length + 1)
        plt.plot(x, means_per_state, color=colors[count2], label=f"{loop_array[count2]}")
        plt.plot((x[0], x[mc_length]), (2569, 2569), 'k-')

        count2 += 1
    
    plt.grid()
    plt.legend()
    plt.xlabel("State of the Markov Chain")
    plt.ylabel("Average route length")
    plt.savefig(f"images/{loop_var}/means_all_methods_{n}_{mc_length}.png")
    plt.show()

# Plot evolution of route length over markov chain states for all four methods without confidence intervals
def plot_per_method(length_dict, mc_length, n, T0, method, loop_var):
    means_per_state = np.ones(mc_length + 1)
    interval_per_state = np.ones(mc_length + 1)

    count1 = 0
    for state in length_dict.items():
        average_length = np.mean(state[1])
        means_per_state[count1] = average_length

        std = np.mean(state[1])
        interval = 1.96 * std / (n ** (1/2))
        interval_per_state[count1] = interval

        count1 += 1

    x = np.linspace(0, mc_length + 1, mc_length + 1)
    plt.plot(x, means_per_state, color='blue', label="Mean")
    plt.plot(x, means_per_state + interval, color='red', label = 'Upper bound')
    plt.plot(x, means_per_state - interval, color='red', label = 'Lower bound')
    plt.plot((x[0], x[mc_length]), (2569, 2569), 'k-')
    plt.fill_between(x, means_per_state + interval, means_per_state - interval, color = 'lightcoral')
    
    plt.grid()
    plt.legend()
    plt.xlabel("State of the Markov Chain")
    plt.ylabel("Average route length")
    plt.savefig(f"images/{loop_var}/mean_interval_{method}_{n}_{mc_length}.png")
    plt.show()

# plot the best route according to a method
def plot_route(route, data, method, loop_var):
    nodes = getSeperateIntCoords(data)

    x = np.array([])
    y = np.array([])

    for city in route:
        x = np.append(x, nodes[int(city) - 1][1])
        y = np.append(y, nodes[int(city) - 1][2])
    x = np.append(x, nodes[int(route[0]) - 1][1])
    y = np.append(y, nodes[int(route[0]) - 1][2])

    plt.plot(x, y, '-o')
    for i, txt in enumerate(route):
        plt.annotate(int(txt), (x[i], y[i]), fontsize='x-small')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"images/{loop_var}/best_route_{method}_{n}_{mc_length}.png")
    plt.show()


############################################# Function calls

# call all functions to generate data and create plots
def iterate(data, n, mc_length, T0, alpha1, alpha2, b, cooling_scheme, loop_array, loop_var):
    loop_dict = {}
    best_route_dict = {}

    for element in loop_array:
        if loop_var == 'cooling_scheme_values':
            all_routes, length_dict, best_route, distance_br = sim(data, n, mc_length, T0, alpha1, alpha2, b, element)
        elif loop_var == 'mc_length': 
            all_routes, length_dict, best_route, distance_br = sim(data, n, element, T0, alpha1, alpha2, b, cooling_scheme)
        else: 
            all_routes, length_dict, best_route, distance_br = sim(data, n, mc_length, element, alpha1, alpha2, b, cooling_scheme)
        loop_dict[element] = length_dict
        best_route_dict[element] = best_route

    if loop_var == 'cooling_scheme_values':
        count = 0
        for element in loop_dict.items():
            plot_per_method(element[1], mc_length, n, T0, loop_array[count], loop_var)
            count += 1

        plot_means(loop_dict, mc_length, n, T0, loop_array, loop_var)

        count = 0
        for element in best_route_dict.items():
            plot_route(element[1], data, loop_array[count], loop_var)
            count += 1

    elif loop_var == 'mc_length':
        count = 0
        for element in loop_dict.items():
            plot_per_method(element[1], loop_array[count], n, T0, cooling_scheme, loop_var)
            count += 1

        count = 0
        for element in best_route_dict.items():
            plot_route(element[1], data, loop_array[count], loop_var)
            count += 1

    else:
        count = 0
        for element in loop_dict.items():
            plot_per_method(element[1], mc_length, n, T0, loop_array[count], loop_var)
            count += 1

        plot_means(loop_dict, mc_length, n, T0, loop_array, loop_var)

        count = 0
        for element in best_route_dict.items():
            plot_route(element[1], data, loop_array[count], loop_var)
            count += 1

if loop_var == 'cooling_scheme_values':
    iterate(data2, n, mc_length, T0, alpha1, alpha2, b, cooling_scheme_values, cooling_scheme_values, loop_var)
elif loop_var == 'mc_length':
    iterate(data2, n, mc_length, T0, alpha1, alpha2, b, cooling_scheme_values, mc_length, loop_var)
else:
    iterate(data2, n, mc_length, T0, alpha1, alpha2, b, cooling_scheme_values, T0, loop_var)