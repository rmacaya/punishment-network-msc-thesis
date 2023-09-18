from functions import *

run(n = 1000,
    m = 2,
    steps = 20000,
    steps_avg = 0.25,
    r = 1.25,
    c = 1,
    network = 0,
    fermi_temp = 5,
    mu = 0.0001,
    punishment_cost = 0.5,
    punishing_cost = 0.2,
    reduction_mechanism = 0,
    reduction_fraction = 0,
    fixed = 1,
    hubs_amount = 5,
    n_experiments = 300,
    experiments_per_seed = 1,
    origin = [],
    ending = [])


### PARAMETER GUIDE ###

#n                      Number of nodes in the population
#m                      m+2 is the average degree of nodes in Scale-free, and m+2 the uniform degree of nodes in Regular graphs
#steps                  Total number of generations
#steps_avg              Proportion of final generations over which we average the cooperation result
#r                      enhancement factor
#c                      Cost of every game in non-fixed mode, total cost a node can afford in fixed mode (c/degree+1 is the cost per game)
#network                0 is a scale-free network, 1 is a watts strogatz regular network
#fermi_temp             Temperature parameter for the Fermi probability distribution (higher temp, lower chance of unfavourable switch)
#mu                     Mutation probability for each node at each iteration
#fixed                  1 = fixed, an individual pays c in total, divided between each game they play
#                        0 = not fixed, an individual pays c in each game they play
#punishment_cost        Fine imposed to Ds by surrounding Ps
#punishing_cost         Cost of punishing with respect to the fines imposed
#reduction_mechanism    0: disabled (full pnetwork), 1: mechanism A, 2: mechanism B, 3: random deletion, 4: social class to social class
#reduction_fraction     For mechanisms A and B: How many high-degree (A) or low-degree (B) nodes are deactivated in the punishment network (in absolute number)
#                       For mechanism C: fraction of the total links to be deleted from the punishment network
#hubs_amount            Top connected hubs of which to track the strategy
#n_experiments          Number of experiments to average from
#experiments_per_seed   Number of experiment to conduct with each randomly generated network

#origin                 List of the origin end-point classes whose outgoing links must be deleted (e.g. if we want to delete links from 3 to 1 and 2: [3,3])
#ending                 For the prior classes of links to be deleted, the corresponding end-point classes (e.g. if we want to delete links from 3 to 1 and 2: [1,2])



