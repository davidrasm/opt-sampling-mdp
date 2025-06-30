# Optimizing genomic sampling with MDPs

For more information, please see the our manuscript:

Rasmussen et al. (2025) Optimizing genomic sampling for demographic and epidemiological inference with Markov decision processes. in prep.

## Optimization code

Markov decision process (MDP) models are implemented for three common scenarios in population genomics and genomic epidemiology:
- **exp_coal_mdp.py** implements the exponential growth coalescent MDP for optimizing sampling to estimate population growth rates. 
- **ttd_mdp.py** implements the transmission tree distance MDP to minimize transmission distances between hosts in a transmission tree.
- **struct_coal_mdp.py** implements the structured coalescent MDP to optimize sampling to estimate migration rates between sub-populations. 

Each MDP model contains high-level functions for optimizing sampling strategies using dynamic programming, including:
- An **eval_policy()** function to compute the long-term expected reward or value of a given sampling policy using a policy iteration algorithm.
- An **opt_policy()** function to optimize sampling policies using a value iteration algorithm.
- A **brute_force()** function to perform a brute force search over policy space to find an optimal policy through Monte Carlo simulations of the MDP.

Note: environment.yml can be used to create a conda or mamba environment with tskit and other required packages installed:
```
$ conda env create -f environment.yml
```
