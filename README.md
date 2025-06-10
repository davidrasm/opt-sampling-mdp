# Optimizing genomic sampling with MDPs

For more information, please see the our manuscript:

Rasmussen et al. (2025) Optimizing genomic sampling for demographic and epidemiological inference with Markov decision processes. in prep.

## Optimization code

Markov decision process (MDP) models are implemented for three common scenarios in populuation genomics and genomic epidemiology:
-The exponential growth coalescent MDP for optimizing sampling to estimate population growth rates in **exp_coal_mdp.py**
-The transmission tree distance MDP to minimize distances between sampled individuals in **ttd_mdp.py**
-The structured coalescent MDP to optimize sampling to estimate migration rates in **struct_coal_mdp.py**

Each MDP model contains high-level functions for optimizing sampling strategies, including:
-An **eval_policy()** function for computing the long-term expected reward/value of a given sampling policy using a policy iteration algorithm.
-An **opt_policy()** function for optimizing sampling policies using a value iteration algorithm.
-A **brute_force()** which performs a brute force search over policy space to find an optimal policy.

Note: environment.yml can be used to create a conda or mamba environment with tskit and other required packages installed:
```
$ conda env create -f environment.yml
```
