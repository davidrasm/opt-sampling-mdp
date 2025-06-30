# Optimizing genomic sampling with MDPs

Project code for optimizing genomic sampling for demographic and epidemiological inference using a sequential decision making framework. Markov decision processes (MDPs) are used to compute the expected rewards in terms of information gained from sampling and optimal strategies are identified using dynamic programming based on the expected rewards. 

For more information, please see our manuscript:

Rasmussen et al. (2025) Optimizing genomic sampling for demographic and epidemiological inference with Markov decision processes. in prep.

## MDP and optimization code

MDPs are implemented for three common scenarios in population genomics and genomic epidemiology:
- **exp_coal_mdp.py** implements the exponential growth coalescent MDP for optimizing sampling to estimate population growth rates. 
- **ttd_mdp.py** implements the transmission tree distance MDP to minimize transmission distances between hosts in a transmission tree.
- **struct_coal_mdp.py** implements the structured coalescent MDP to optimize sampling to estimate migration rates between sub-populations. 

Each MDP model contains high-level functions for optimizing sampling strategies using dynamic programming, including:
- An **iter_eval_policy()** function to compute the long-term expected reward or value of a given sampling policy using an iterative policy evaluation algorithm.
- An **opt_policy()** function to optimize sampling policies using a value iteration algorithm.
- a **q_value()** to compute expected value (q-value) of a given state-action pair under a policy
- A **brute_force()** function to perform a brute force search over policy space to find an optimal policy through Monte Carlo simulations of the MDP.

## Parameter estimation

Likelihood functions are also provided for maximum likelihood parameter estimation:
- **exp_coal_like.py** computes the likelihood of a tree under the exponential growth coalescent model of Kuhner et al. (Genetics, 1998)
- **struct_coal_like.py** computes the likelihood of a tree under the approximations to the structured coalescent model in Volz (Genetics, 2012) and Müller et al. (MBE, 2017).

Note: environment.yml can be used to create a conda or mamba environment with the required Python packages installed:
```
$ conda env create -f environment.yml
```
