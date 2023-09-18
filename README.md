# Public Goods Games played over a network of contacts, with peer-punishment exerted through a directed punishment network.

### This code is the implementation of the model in the MSc Thesis "Dynamics of peer-monitoring and sustainable action in complex networks" by Roger Macaya Munell,  in fulfilment of the requirements for the joint UvA-VU Master of Science degree in Physics and Astronomy (Science for Energy and Sustainability track).

It consists of two files "run.py" and "functions.py".
To run a series of experiments, open run.py and input the parameters onto the run() function. You can easily run several series of experiments with different parameters by calling multiple instances of run().
The file functions.py is where the simulations take place.

A quick explanation on the parameters (although best is to check the full document on the MSc Thesis):
| Parameter  | Explanation |
| ------------- | ------------- |
| **n**  | Number of nodes in the population  |
| **m**  | Links to be added at each step of the Barabási-Albert algorithm for the generation of the scale-free network. Therefore, m+2 is the average degree of nodes in Scale-free. In case of a regular network, m+2 the uniform degree of nodes in Watts strogatz grpahs  |        
| **steps** | Total number of generations  |
| **steps_avg**  | Proportion of final generations over which we average the cooperation result  |
| **r**  |                       enhancement factor  |
| **c**  |                       Cost of every game in non-fixed mode, total cost a node can afford in fixed mode (c/degree+1 is the cost per game)  |
| **network**  |                 0 is a scale-free network, 1 is a watts strogatz regular network. (Note that it is extremely easy to implement other networks on functions.py thanks to the NetworkX generation)  |
| **fermi_temp**  |              Temperature parameter for the Fermi probability distribution (higher temp, lower chance of unfavourable switch)  |
| **mu**  |                      Mutation probability for each node at each iteration  |
| **fixed**  |                   1 = fixed, an individual pays c in total, divided between each game they play. 0 = not fixed, an individual pays c in each game they play  |
| **punishment_cost**  |         Fine imposed to Ds by surrounding Ps  |
| **punishing_cost**  |          Cost of punishing with respect to the fines imposed  |
| **reduction_mechanism**  |     0: disabled (full pnetwork), 1: mechanism A, 2: mechanism B, 3: random deletion, 4: social class to social class  |
| **reduction_fraction**  |      For mechanisms A and B: How many high-degree (A) or low-degree (B) nodes are deactivated in the punishment network (in absolute number). For mechanism C: fraction of the total links to be deleted from the punishment network. Has no effect with mechanism 4 (D), see origin and ending below.  |
| **hubs_amount**  |             Top connected hubs of which to track the strategy  |
| **n_experiments**  |           Number of experiments to average from  |
| **experiments_per_seed**  |    Number of experiment to conduct with each randomly generated network  |
| **origin**  |                  (For mechanism 4 (D)): List of the origin end-point classes whose outgoing links must be deleted (e.g. if we want to delete links from 3 to 1 and 2: [3,3])  |
| **ending**  |                  (For mechanism 4 (D)): For the prior classes of links to be deleted, the corresponding end-point classes (e.g. if we want to delete links from 3 to 1 and 2: [1,2])  |
