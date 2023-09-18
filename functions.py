import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
from numba import njit, prange
from numba.typed import List
from datetime import datetime
import time
import os

def run(n, m, steps, steps_avg, r, c, network, fermi_temp, mu, punishment_cost, punishing_cost, reduction_mechanism, reduction_fraction, fixed, hubs_amount, n_experiments, experiments_per_seed, origin, ending):

    st = time.time()            # Starting timestamp
    pt = time.time()            # Previous experiment timestamp (we initialize it at 0)

    results = []                # List where the fraction of cooperation for each experiment will be stored
    results_pun = []            # List where the fraction of punishers will be stored
    conv_step = []              # List where we store the convergence step of each experiment (in case we apply convergence conditions)

    network_seed = []           # List that will contain the random seed for the generation of each random network at each experiment

    for j in range(n_experiments // experiments_per_seed):      # We generate the requiried random seeds
        network_seed.append(np.random.randint(0,1000000))


    now = datetime.now()        # We store the datetime for the folder and filenames

    if network == 0:
        network_type = "Scale-free Graph"
    elif network == 1:
        network_type = "Regular Graph"
    else:
        print("Network needs to be 0 (scale-free) or 1 (regular)!")

    directory = str(now.strftime("%d-%m-%Y %H.%M.%S"))      # The directory where the plots and textfiles will be stored for this set of experiments

    os.mkdir(directory)

    txt = open(str(directory)+'/results.txt', 'a')                      # We store the parameters in a txt file
    txt.write(directory+"\n\n")
    txt.write("r = " + str(r) + "\n")
    txt.write(str(network_type) + "\n")
    txt.write("list of seeds = " + str(network_seed) + "\n")
    txt.write("n = " + str(n) + "\n")
    txt.write("steps = " + str(round(steps/1000)) + "k \n")
    txt.write("fixed per game? = " + str(fixed) + "\n")
    txt.write("T fermi = " + str(fermi_temp) + "\n")
    txt.write("mu (mutation probability) = " + str(mu) + "\n")
    txt.write("Punishment cost = " + str(punishment_cost) + "\n")
    txt.write("Relative punishing cost = " + str(punishing_cost) + "\n")
    txt.write("Reduction mechanism = " + str(reduction_mechanism) + "\n")
    if reduction_mechanism==1 or reduction_mechanism==2 or reduction_mechanism==3:
        txt.write("Fraction of punishment links to be deleted = " +str(reduction_fraction))
    elif reduction_mechanism==4:
        txt.write("Deleting all links from social class: " + str(origin) + " to social class: " + str(ending) + ". \n")


    for i in range(n_experiments):                                  # For each experiment...
        if network == 0:                                            # ... we generate a network
            Graph = nx.barabasi_albert_graph(n, m, network_seed[i // experiments_per_seed])

        elif network == 1:
            Graph = nx.watts_strogatz_graph(n, m+2, 0)

                                                                    # ... we call simulate, which starts the experiment
        cooperators, punishers_data, convergence_step, hubs_changes, hubs_degree, avg_fitness = simulate(Graph, steps, c, r, fixed, fermi_temp, mu, punishment_cost, punishing_cost, reduction_mechanism, reduction_fraction, hubs_amount, origin, ending)

        txt.write("\nExperiment "+str(i)+":\n Cooperators = "+str(cooperators[convergence_step])+"("+str(100*cooperators[convergence_step]/n)+"%)\nP = "+str(punishers_data[convergence_step])+"("+str(100*punishers_data[convergence_step]/n)+"%)\n")
        txt.write("Seed: " + str(network_seed[i // experiments_per_seed]) + "\n")

        iterations_averaged = round(convergence_step*steps_avg)     # ... we calculate after which generation to average the results for

        if convergence_step != steps:                               # ... if we imposed convergence conditions and it converged earlier
            results.append(cooperators[convergence_step])
            results_pun.append(punishers_data[convergence_step])
        else:
            results.append(np.round(np.average(cooperators[-iterations_averaged:]), 2))
            results_pun.append(np.round(np.average(punishers_data[-iterations_averaged:]), 2))

        conv_step.append(convergence_step)

                                                                    # We plot the evolution of the strategies and hubs strategies with "grap_pun_hubs_sublot"
        graph_pun_hubs_subplot(i,convergence_step, cooperators, punishers_data, iterations_averaged, n, m, steps, r, directory, network_type, hubs_changes, hubs_degree)

        print("Experiment " + str(i) + ". Cooperators: " + str(np.round(np.average(cooperators[-iterations_averaged:]), 2))+" in "+str(convergence_step + 1)+" iterations.")
        print("Time elapsed = " + time.strftime("%H:%M:%S", time.gmtime(time.time() - pt)))
        print("Seed: "+str(network_seed[i // experiments_per_seed]))

        #graph_avg_fitness(i, convergence_step, avg_fitness, directory)     # By uncommenting this we can plot the average fitnesses for each strategy

        pt = time.time()

    elapsed_time = time.time() - st

    print("Average cooperators: " + str(np.average(results)))
    print("Average punishers: " + str(np.average(results_pun)))
    print("Average convergence step: " + str(np.average(conv_step)))
    print("Number of experiments with C = 0: " +str(np.count_nonzero(results == 0)))
    print("Number of experiments that converged: " + str(np.count_nonzero(conv_step != steps-1)))
    print("\nExecution time: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    txt.write("\n\nAverage cooperators: " + str(np.average(results)))
    txt.write("\nAverage punishers: " + str(np.average(results_pun)))
    txt.write("\nAverage convergence step: " + str(np.average(conv_step)))
    txt.write("\nNumber of experiments with C = 0: " +str(np.count_nonzero(results == 0)))
    txt.write("\nNumber of experiments that converged: " + str(np.count_nonzero(conv_step != steps-1)))
    txt.write("\n\nExecution time: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    txt.write("\nTime per experiment: "+time.strftime("%H:%M:%S", time.gmtime(elapsed_time/n_experiments)))
    txt.write('\nTime per 1k iterations: ' + str(elapsed_time*1000/np.sum(conv_step)) + " seconds")
    txt.close()

    os.rename(directory, directory+', C='+str(100*np.average(results)/n)+'%')           # We rename directory to include final cooperation


def simulate (Graph, cycles, c, r, fixed, fermi_temp, mu, punishment_cost, punishing_cost, reduction_mechanism, reduction_fraction, hubs_amount, origin, ending):

    neighbors_list = []                          # List of lists: for each node a list of the neighbors

    degree_list = []                             # List of degrees of each node

    cost_list = []                               # List of the costs for each node (c if fixed per game, c/k+1 if fixed per individual)

    global G
    G = Graph


    size = Graph.number_of_nodes()

    strategies = np.zeros(size)               # List of current strategies for each node. 0 is cooperate, 1 is defect
    punishers = np.zeros(size)                # List that distinguishes cooperators (0) from punishers (1)

    for node in prange(size):                 # We initialize the strategies for each node randomly, 50% D, 25% C, 25% P
        neighbors_list.append([])
        strat = np.random.randint(0,4)
        if strat == 0: strategies[node] = 1
        elif strat == 1: strategies[node] = 1
        elif strat == 2:
            strategies[node] = 0
            punishers[node] = 1
        elif strat == 3:
            strategies[node] = 0
            punishers[node] = 0

        for pair in Graph.edges(node):                      # For each neighbor of the node we are iterating over...
            neighbors_list[node].append(pair[1])            # ...we create and append a list of neighbors

        degree_list.append(Graph.degree[node])

    # STORE LISTS INTO NP ARRAYS:

    neighbors = List()                                      # neighbors is a Numba typed List()
    [neighbors.append(np.asarray(e, dtype=np.int64)) for e in neighbors_list]   # neighbors is a numba List() of np.arrays with the neighbors index (int64) of each node
    degree = np.asarray(degree_list)                        # We convert our degree_list to a numpy array for Numba

    # PUNISHMENT NETWORK:
    pnetwork = List()                                       # The punishment network including the directed links will be stored in a Numba List()
    social_dist = List()                                    # List() to classify the nodes in social classes

    banned = np.zeros(size)                                 # Here we will store the deactivated nodes if reduction mechanism is A or B

    ### DELETING LINKS FROM THE PUNISHMENT NETWORK ###

    # If mechanism is 0: no deletion, full pnetwork.
    if reduction_mechanism == 0:
        for node in range(size):
            pnetwork.append(np.asarray(neighbors[node]))    # Punishment network is a copy of game network with bi-directed links

    # If mechanism is 1 (A): deleting outgoing links from hubs
    if reduction_mechanism == 1:
        banned_index = np.argpartition(degree, reduction_fraction)[-reduction_fraction:]     # We select the indices of the desired number of nodes with highest degree
        banned[banned_index] = 1
        for node in range(size):
            if banned[node] == 1:
                pnetwork.append(np.asarray([], dtype=np.int64))
            if banned[node] == 0:
                pnetwork.append(np.asarray(neighbors[node]))


    # If mechanism is 2 (B): deleting outgoing links from low-degree nodes
    if reduction_mechanism == 2:
        banned_index = np.argpartition(degree, reduction_fraction)[:reduction_fraction]           # We select the indices of the desired number of nodes with lowest degree
        banned[banned_index] = 1
        for node in range(size):
            if banned[node] == 1:
                pnetwork.append(np.asarray([], dtype=np.int64))
            if banned[node] == 0:
                pnetwork.append(np.asarray(neighbors[node]))

    # If mechanism is 3 (C): deleting random links
    if reduction_mechanism == 3:
        for node in range(size):
            pnetwork.append(np.asarray(neighbors[node]))
        pnetwork = remove_random_links(reduction_fraction, degree, pnetwork, size)

    # If mechanism is 4: deleting all links from one social class to another
    if reduction_mechanism == 4:
        social_dist.append(np.argwhere(degree == 2))                            # Segment 0: 2
        social_dist.append(np.argwhere((degree == 3) | (degree == 4)))          # Segment 1: 3-4
        social_dist.append(np.argwhere((degree >= 5) & (degree <= 10)))         # Segment 2: 5-10
        social_dist.append(np.argwhere((degree >= 11) & (degree <= 20)))        # Segment 3: 11-20
        social_dist.append(np.argwhere(degree > 21))                            # Segment 4: >21

        for node in range(size):
            pnetwork.append(np.asarray(neighbors[node]))
            for k in range(len(origin)):
                if node in social_dist[origin[k]]:
                    i = 0
                    kill = List()
                    for neighbor in pnetwork[node]:
                        if neighbor in social_dist[ending[k]]:
                            kill.append(i)
                        i += 1
                    pnetwork[node] = np.delete(pnetwork[node], kill)

    ### IS COST FIXED PER GAME (0) OR FIXED PER INDIVIDUAL (1) ###
    if fixed == 0:                          # If cost is fixed per game, every node pays c in each game
        for node in range(size):
            cost_list.append(c)

    elif fixed == 1:                        # If cost is fixed, every node pays c/(degree+1) in each of their games
        for node in range(size):
            cost_list.append(c / (degree[node] + 1))

    cost = np.asarray(cost_list)

    ### HUBS STRATEGY TRACKING ###
    hubs = np.argpartition(degree, -hubs_amount)[-hubs_amount:]         # We find the indices of the nodes that correspond to the "hubs_amount" biggest hubs

    hubs_degree = []
    for hub in hubs:
        hubs_degree.append(degree[hub])                                 # We store their degree

    ### WE RUN THE EXPERIMENT ###
    cooperators, punishers_data, new_strategies, new_fitness, convergence_step, hubs_changes, avg_fitness = iterate(neighbors, degree, punishers, pnetwork, strategies, cost, cycles, r, size, fermi_temp, mu, punishment_cost, punishing_cost, hubs)
    return cooperators, punishers_data, convergence_step, hubs_changes, hubs_degree, avg_fitness

@njit(fastmath = True)
def iterate(neighbors, degree, punishers, pnetwork, strategies, cost, cycles, r, size, fermi_temp, mu, punishment_cost, punishing_cost, hubs):

    cooperators = np.zeros(cycles)              # Here we store the number of cooperators at each generation
    punishers_data = np.zeros(cycles)           # Here we store the number of punishers at each generation
    avg_fitness = np.zeros((3, cycles))         # Here we store the average fitness at each generation for each strategy

    hubs_changes = List()                       # List() that stores the strategy updates (by imitation or mutation) of the hubs
    for hub in hubs:
        hubs_changes.append([[strategies[hub]+2*punishers[hub],0,0]])


    for iteration in range(cycles):                 # AT EACH GENERATON:

        total_def = np.sum(strategies)                  # Number of defectors
        total_coop = strategies.size - total_def        # Number of C + P
        cooperators[iteration] = total_coop             # We store cooperation
        punishers_data[iteration] = np.sum(punishers)   # We store P fraction

        # GAMES ARE PLAYED and PAYOFFS ARE COMPUTED

        fitness = np.zeros(size)                        # The fitness is reset every generation

        for node in range(size):                                                ####### FOR EACH PGG ########
            neighbors_strat_inv = 1-strategies[np.asarray(neighbors[node])]     # Array with 1s for cooperatng neigbors and 0s for defecting neighbors

            total_cost = 0      # Computes the total contributions
            for i in range(degree[node]):                                                                               # We do it in a for loop instead of np.dot because numba doesnt support dot product for integers
                total_cost += cost[neighbors[node]][i]*neighbors_strat_inv[i]                                           # The total contribution is the sum of each neighbors cost*inv_strat + own node's
            total_cost += cost[node]*(1-strategies[node])
            payoff_defect = total_cost*r/(degree[node]+1)                                                               # Payoff for the defectors (payoff for the cooperators will be this minus their cost)

            if strategies[node] == 1:                                                                                   # If the node is a defector...
                fitness[node] += payoff_defect                                                                          # It gains payoff_defect
                if np.sum(punishers[neighbors[node]]) > 0:                                                              # If it has punishers around
                    for pun in check_punishers(node, node, neighbors[node], neighbors[node], punishers, pnetwork):      # We check how many punishers are connected to them...
                        fitness[node] += - punishment_cost                                                                  # The node is punished for each of them
                        fitness[pun] += - punishment_cost * punishing_cost                                                 # Each punisher pays the price to punish
            elif strategies[node] == 0:                                                                                 # If the node is a C, it gains the payoff minus the cost
                fitness[node] += payoff_defect - cost[node]

            for neighbor in neighbors[node]:                                                                            # For each neighbor of the center node...
                if strategies[neighbor] == 0:                                                                           # If it is a C... it gains payoff - cost
                    fitness[neighbor] += payoff_defect - cost[neighbor]
                if strategies[neighbor] == 1:                                                                           # If it is a D... it gains payoff
                    fitness[neighbor] += payoff_defect
                    if np.sum(punishers[neighbors[neighbor]]) > 0:                                                      # We check how many punishers are connected to them...
                        for pun in check_punishers(neighbor, node, neighbors[neighbor], neighbors[node], punishers, pnetwork):
                            fitness[neighbor] += - punishment_cost                                                          # The node is punished for each of them
                            fitness[pun] += - punishment_cost * punishing_cost                                             # Each punisher pays the price to punish

        #TRACKING THE AVERAGE FITNESSES
        C_total_fitness = 0
        D_total_fitness = 0
        P_total_fitness = 0
        for node in prange(size):
            if strategies[node] == 0 and punishers[node] == 0 : C_total_fitness += fitness[node]
            if strategies[node] == 0 and punishers[node] == 1: P_total_fitness += fitness[node]
            if strategies[node] == 1: D_total_fitness += fitness[node]
        if total_coop != 0 : avg_fitness[0][iteration] = C_total_fitness/total_coop                                     # ... so as to not divide by 0
        else : avg_fitness[0][iteration] = 0
        if total_def != 0 : avg_fitness[1][iteration] = D_total_fitness/total_def
        else : avg_fitness[1][iteration] = 0
        if punishers_data[iteration] != 0 : avg_fitness[2][iteration] = P_total_fitness/punishers_data[iteration]
        else : avg_fitness[2][iteration] = 0

        ### UPDATING STRATEGIES (MUTATION AND IMITATIONS) ###
        Old_strategies = np.copy(strategies)                                # We store the strategies to be able to do a synchronous update
        Old_punishers = np.copy(punishers)

        for i in range(size):                                               # For each node...
            node = i
            chosen_neighbor = np.random.randint(0, degree[node])            # ... Choose a neighbor at random
            chosen_node = neighbors[node][chosen_neighbor]

            seed = np.random.rand()
            if seed < mu:                                                   # Does a mutation occur?
                seed2 = np.random.randint(0,3)
                if seed2 == 0:                                             # 33% it mutates into a Defector
                    strategies[node] = 1
                    punishers[node] = 0
                elif seed2 == 1:                                           # 33% it mutates into a Cooperator
                    strategies[node] = 0
                    punishers[node] = 0
                elif seed2 == 2:                                           # 33% it mutates into a Punisher
                    strategies[node] = 0
                    punishers[node] = 1
                if node in hubs:                                            # If the node is a main hub, we take note of the mutation
                    hubs_changes[np.where(hubs==node)[0][0]].append([strategies[node]+2*punishers[node], iteration, 1])

            # IF THERE IS NO MUTATION, THEN IMITATE:
            else:
                if Old_strategies[chosen_node] != strategies[node]:                                                     # If strategies are different (C/P vs D)
                    probability = 1 / (1 + np.exp(fermi_temp * (fitness[node] - fitness[chosen_node])))                 # Fermi Update Rule gives us an imitation probability
                    seed = np.random.rand()
                    if seed < probability:                                                                              # Check if seed is within generated probability
                        strategies[node] = Old_strategies[chosen_node]
                        punishers[node] = Old_punishers[chosen_node]
                        if node in hubs:                                                                                # If the node is a main hub, we take note of the strategy update
                            hubs_changes[np.where(hubs==node)[0][0]].append([strategies[node]+2*punishers[node], iteration, 0])

                elif Old_strategies[chosen_node] == strategies[node] and Old_punishers[chosen_node] != punishers[node]: # If both are cooperators but only one is punisher:
                    probability = 1 / (1 + np.exp(fermi_temp * (fitness[node] - fitness[chosen_node])))
                    seed = np.random.rand()
                    if seed < probability:
                        punishers[node] = Old_punishers[chosen_node]
                        if node in hubs:
                            hubs_changes[np.where(hubs==node)[0][0]].append([strategies[node]+2*punishers[node], iteration, 0])

    convergence_step = iteration

    return (cooperators, punishers_data, strategies, fitness, convergence_step, hubs_changes, avg_fitness)

def remove_random_links(proportion, degree, pnetwork, size):

    total_links = np.sum(degree)
    amount = np.round(proportion * total_links)

    for i in range(np.int(amount)):
        weights = [j.size for j in pnetwork]
        weights_sum = sum(weights)
        weights_norm = [k/weights_sum for k in weights]
        selected_node = np.random.choice(np.arange(size), p = weights_norm)
        pnetwork[selected_node] = np.delete(pnetwork[selected_node], np.random.randint(0, pnetwork[selected_node].size))
    return pnetwork

@njit
def check_punishers (node, center_node, neighbors_node, neighbors_center_node, punishers, pnetwork):
    #node_punishers = List()
    group = np.intersect1d(neighbors_node, neighbors_center_node)
    n_punishers = np.zeros(group.size, dtype=np.int64)
    i = 0
    for neighbor in group:                        # We go through common neighbors (from the node in question and the central node) plus the center_node
        if punishers[neighbor] == 1 and node in pnetwork[neighbor]:
            n_punishers[i] = neighbor
            #node_punishers.append(neighbor)
            i += 1
    if punishers[center_node] == 1 and node in pnetwork[center_node]:
        n_punishers[i] = center_node
        #node_punishers.append(center_node)
        i += 1
    return n_punishers[:i]
    #return node_punishers

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

def graph_cooperation_time(iteration,convergence_step, cooperators, n, k, steps, r, directory, network_type):

    x_coordinates = np.arange(0, convergence_step + 1)
    y_coordinates = cooperators[0:convergence_step + 1]
    # print("Convergence step: "+str(convergence_step))
    # print("Cooperators array length: " + str(len(cooperators)))
    plt.plot(x_coordinates, y_coordinates, '-o')
    plt.xlabel("Time (iteration)")
    plt.ylabel("Cooperators (%)")
    plt.title(network_type+', n='+str(n)+', k='+str(k)+', steps='+str(steps)+', r='+str(r))

    plt.savefig(str(directory)+'/'+str(iteration)+' cooperators='+str(cooperators[convergence_step])+' conv_step='+str(convergence_step)+'.png')

    plt.close()

def graph_punishers_time(iteration,convergence_step, cooperators, punishers, iterations_averaged, n, k, steps, r, directory, network_type):

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    font = {'family': "serif",
            'serif': ["Computer Modern Serif"],
            'size': 12}

    mpl.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))

    x_coordinates = np.arange(0, convergence_step + 1)
    y_c = cooperators[0:convergence_step + 1] - punishers[0:convergence_step + 1]
    y_p = punishers[0:convergence_step + 1]
    y_d = n - cooperators[0:convergence_step + 1]
    y_half = n/2

    ax1.plot(x_coordinates, y_c, '-', label = "C")
    ax1.plot(x_coordinates, y_p, '-m', label = "P")
    ax1.plot(x_coordinates, y_d, '-r', label = "D")
    ax1.plot(x_coordinates, cooperators[0:convergence_step + 1], '-g', label = "C + P")
    ax1.axhline(y_half, linestyle='dotted', color="k" )
    ax1.set_xlabel("Time (iteration)")
    ax1.set_ylabel("Number of individuals")
    ax1.legend(loc = "upper right")
    #plt.title(network_type+', n='+str(n)+', k='+str(k)+', steps='+str(steps)+', r='+str(r))
    # fig.tight_layout()
    plt.savefig(str(directory)+'/'+str(iteration)+' C+P='+str(np.round(np.average(cooperators[-iterations_averaged:]), 2))+' P='+str(np.round(np.average(punishers[-iterations_averaged:]), 2))+' Step='+str(convergence_step)+'.png', dpi=200)

    plt.close()

def graph_hubs(iteration,convergence_step, cooperators, n, k, steps, r, directory, network_type, hubs_changes, hubs_degree):

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    font = {'family': "serif",
            'serif': ["Computer Modern Serif"],
            'size': 12}

    mpl.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))

    x_coordinates = np.arange(0, convergence_step + 1)
    j = 0
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for hub in hubs_changes:
        history = []
        mutations = []
        index = 0
        for period in hub:                                                                      # Every period has the structure [new strategy, iteration, mutation? (1 or 0)]
            if index != 0:
                #for it in range(np.int(period[1])-len(history)+1):                                  # We append y points for the current strategy for the length of this
                while len(history) < np.int(period[1]):
                    history.append(previous_strat)
            previous_strat = period[0]
            index += 1
            if period[2] == 1:                                                                   # If this change is a mutation... we append the iteration in which it happens
                mutations.append(np.int(period[1]))

        #for it in range(convergence_step-np.int(hub[-1][1])):                                   # When we have gone through all the "periods" then we fill in the rest of the iterations with the last strat
        while len(history) < len(x_coordinates):
            history.append(hub[-1][0])
        history2 = [x + 0.03*j for x in history]
        ax1.plot(x_coordinates, history2, color=color_cycle[j], linestyle='dashed', alpha=0.5, label = "Degree: "+str(hubs_degree[j]))
        for mutation in mutations:
            x_mut = [mutation-1, mutation]
            y_mut = [history2[mutation-1], history2[mutation]]

            xpos = (x_mut[0] + x_mut[1]) / 2
            ypos = (y_mut[0] + y_mut[1]) / 2
            xdir = x_mut[1] - x_mut[0]
            ydir = y_mut[1]- y_mut[0]

            ax1.plot(x_mut, y_mut, color=color_cycle[j], linestyle='solid', alpha=0.9)
            ax1.annotate("", xytext=(xpos,ypos),xy=(xpos+0.001*xdir,ypos+0.001*ydir), arrowprops=dict(arrowstyle="->", color=color_cycle[j]), size = 20)
        j+=1
    ax1.set_xlabel("Time (iteration)")
    ax1.set_ylabel("Strategies")
    ax1.set_yticks([0, 1, 2], ['C', 'D', 'P'])
    #plt.title("Evolution of the strategies for the "+str(len(hubs_changes))+" biggest hubs over time")
    ax1.legend(loc="best")
    #fig.tight_layout()
    plt.savefig(str(directory)+'/'+str(iteration)+' cooperators='+str(cooperators[convergence_step])+' conv_step='+str(convergence_step)+'.png', dpi=200)
    plt.close()

def graph_avg_fitness(iteration, convergence_step, avg_fitness, directory):

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    font = {'family': "serif",
            'serif': ["Computer Modern Serif"],
            'size': 12}

    mpl.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))

    x_coordinates = np.arange(0, convergence_step + 1)
    y_c = avg_fitness[0]
    y_p = avg_fitness[2]
    y_d = avg_fitness[1]

    ax1.plot(x_coordinates, y_c, '-', label = "C", alpha=0.5)
    ax1.plot(x_coordinates, y_p, '-m', label = "P", alpha=0.5)
    ax1.plot(x_coordinates, y_d, '-r', label = "D", alpha=0.5)
    ax1.set_xlabel("Time (iteration)")
    ax1.set_ylabel("Average fitness")
    ax1.set_ylim(-1, 1)
    ax1.legend(loc = "best")
    #plt.title("Evolution of the average fitness per strategy")
    #fig.tight_layout()
    plt.savefig(str(directory)+'/'+str(iteration)+' fitnesses.png', dpi=200)

    plt.close()

def graph_pun_hubs_subplot(iteration,convergence_step, cooperators, punishers, iterations_averaged, n, m, steps, r, directory, network_type, hubs_changes, hubs_degree):

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    font = {'family': "serif",
            'serif': ["Computer Modern Serif"],
            'size': 12}

    mpl.rc('font', **font)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 5.5), gridspec_kw=dict(hspace=0))
    #fig.subplots_adjust(top=0.80, right=0.9,hspace=0)

    x_coordinates = np.arange(0, convergence_step + 1)
    y_c_total = cooperators[0:convergence_step + 1] - punishers[0:convergence_step + 1]
    y_p_total = punishers[0:convergence_step + 1]
    y_d_total = n - cooperators[0:convergence_step + 1]
    y_cp_total = cooperators[0:convergence_step + 1]

    y_c = [i / 1000 for i in y_c_total]
    y_p = [i / 1000 for i in y_p_total]
    y_d = [i / 1000 for i in y_d_total]
    y_cp = [i / 1000 for i in y_cp_total]


    y_half = n / 2000

    axs[0].plot(x_coordinates, y_c, '-', label="C")
    axs[0].plot(x_coordinates, y_p, '-m', label="P")
    axs[0].plot(x_coordinates, y_d, '-r', label="D")
    axs[0].plot(x_coordinates, y_cp, '-g', label="C + P")
    axs[0].axhline(y_half, linestyle='dotted', color="k")
    axs[0].set_ylabel("Fraction of individuals")
    axs[0].legend(loc = "upper right")

    j = 0
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for hub in hubs_changes:
        history = []
        mutations = []
        index = 0
        for period in hub:  # Every period has the structure [new strategy, iteration, mutation? (1 or 0)]
            if index != 0:
                # for it in range(np.int(period[1])-len(history)+1):                                  # We append y points for the current strategy for the length of this
                while len(history) < np.int(period[1]):
                    history.append(previous_strat)
            previous_strat = period[0]
            index += 1
            if period[2] == 1:  # If this change is a mutation... we append the iteration in which it happens
                mutations.append(np.int(period[1]))

        # for it in range(convergence_step-np.int(hub[-1][1])):                                   # When we have gone through all the "periods" then we fill in the rest of the iterations with the last strat
        while len(history) < len(x_coordinates):
            history.append(hub[-1][0])
        history2 = [x + 0.03 * j for x in history]
        axs[1].plot(x_coordinates, history2, color=color_cycle[j], linestyle='dashed', alpha=0.5,
                 label="Degree: " + str(hubs_degree[j]))
        for mutation in mutations:
            x_mut = [mutation - 1, mutation]
            y_mut = [history2[mutation - 1], history2[mutation]]

            xpos = (x_mut[0] + x_mut[1]) / 2
            ypos = (y_mut[0] + y_mut[1]) / 2
            xdir = x_mut[1] - x_mut[0]
            ydir = y_mut[1] - y_mut[0]

            axs[1].plot(x_mut, y_mut, color=color_cycle[j], linestyle='solid', alpha=0.9)
            axs[1].annotate("", xytext=(xpos, ypos), xy=(xpos + 0.001 * xdir, ypos + 0.001 * ydir),
                         arrowprops=dict(arrowstyle="->", color=color_cycle[j]), size=20)
        j += 1
    axs[1].set_xlabel("Time (iteration)")
    axs[1].set_ylabel("Strategies")
    axs[1].set_yticks([0, 1, 2], ['C', 'D', 'P'])
    axs[1].legend(loc="upper right")

    ax = axs[1].secondary_xaxis('top')
    ax.set_xlim(axs[1].get_xlim())
    ax.set_xticks(axs[1].get_xticks())
    ax.tick_params(labeltop=False)
    fig.tight_layout()
    plt.savefig(str(directory) + '/' + str(iteration) + ' cooperators=' + str(cooperators[convergence_step]) + ' conv_step=' + str(convergence_step) + '.pdf', dpi=500)
    plt.close()