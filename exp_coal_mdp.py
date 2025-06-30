#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:33:28 2024

Exponential coalescent MDP module for optimizing sampling to estimate population growth rates under
the coalescent with exponential growth.

The main functions include:
    iter_policy_eval(): to compute the long-term expected reward or value of a given policy by iterative policy evalution.
    opt_policy(): to optimize sampling policies using by value iteration.
    q_value(): to compute expected value (q-value) of a given state-action pair under a policy
    brute_force_search(): to perform a brute force search over policy space to find optimal policy.

@author: david
"""

import msprime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randrange
import itertools
import exp_coal_like as expcoal

def sim_exp_growth_tree(params,sample_config=None,plot=False):
    
    """
        Simulate under an exponential growth demographic model
        Assumes one isolated haploid population
        See here how to sample serially through time:
            https://tskit.dev/msprime/docs/stable/ancestry.html#sampling-time
        Important: Init pop sizes are sizes at present such that size of population is se^{rt} at time t in the past
    """
    
    init_Ne = params['Ne']
    growth_rate = params['growth_rate']
    demography = msprime.Demography.isolated_model([init_Ne], growth_rate=[growth_rate])
    
    if sample_config:
    
        # Sample at multiple time points
        sample_times, sample_sizes = sample_config
        sample_sets = [msprime.SampleSet(si, time=ti) for ti,si in zip(sample_times, sample_sizes)]
    
        # Sample at one time point
        ts = msprime.sim_ancestry(samples=sample_sets, ploidy=1, demography=demography)
        
    else: 
    
        # Sample at present
        ts = msprime.sim_ancestry(params['samples'], ploidy=1, demography=demography)
    
    if plot:
        print(demography)
        print()
        for tree in ts.trees():
            print("-" * 20)
            print("tree {}: interval = {}".format(tree.index, tree.interval))
            print(tree.draw(format="unicode"))
        print(ts.tables.nodes)
        print()
    
    return ts

def test_pair_coal_times(params):
    
    """
        Compare simulated dist of pairwise coal times against prob density given
        in eq. 5 of Slatkin & Hudson (1991) for an exp growing haploid pop
    """
    
    Ne = params['Ne']  # effective pop sizes
    r = params['growth_rate']
    params['samples'] = 2
    
    "Exact pdf"
    times = params['times'] # grid of times at which to eval pdf
    pdf = (np.exp(r*times) / Ne) * np.exp(-(np.exp(r*times) - 1) / (Ne*r))
    
    "Sim pairwise coal times in msprime"
    sims = 1000
    coal_times = []
    for s in range(sims):
        ts = sim_exp_growth_tree(params,plot=False)
        tree = ts.first()
        coal_times.append(tree.time(tree.root))
        
    f, ax = plt.subplots(figsize=(5,3))
    sns.histplot(x=coal_times, stat='density', kde=True, binwidth=1.0)
    sns.lineplot(x=times, y=pdf, ax=ax, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')


def test_ltt_probs(params):
    
    """
        Test lineage through time (LTT) q_k (p_k here) probs with sequential sampling
    """
    
    times = params['times'] #np.linspace(0, 50, n_times+1)
    n_times = len(times)
    
    "Set up sample config"
    sample_times = [0] #[0,10] # needs to be a subset of times
    sample_sizes = [4] #[3,3]
    sample_config = (sample_times,sample_sizes)
    
    k_max = 4
    params['k_max'] = k_max
    
    "Simulate distribution of LTT"
    sims = 1000
    sim_ltt = np.zeros((k_max+1,n_times,sims),int)
    for s in range(sims):
        ts = sim_exp_growth_tree(params,sample_config=sample_config,plot=False)
        tree = ts.first()
        ltt = np.array([tree.num_lineages(t) for t in times])
        ltt[ltt < 1] = 1 # Set times where k=0 to k=1 to reflect persistance of root lineage
        bit_ltt = np.zeros((k_max+1,n_times),int)
        for i in range(n_times):
            bit_ltt[ltt[i],i] = 1
        sim_ltt[:,:,s] = bit_ltt
    sim_ltt_probs = np.sum(sim_ltt,axis=2) / sims
    
    "Analytically compute LTT (q_k) probs"
    ltt_probs = compute_ltt_probs(times,params,sample_config)
    pk = np.zeros(k_max+1)
    pk[sample_sizes[0]] = 1.0
    ltt_probs[:,0] = pk
    
    "Plot p_k probs by k"
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # for k in range(1,k_max+1):
    #     label = "k=" + str(k)
    #     ax.plot(times, ltt_probs[k,:], linewidth=2.0, label=label)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('p_k')
    # ax.legend()
    
    #sns.set(style="darkgrid")
    cmap = sns.color_palette("colorblind", k_max)
    
    "Plot simulated p_k probs by k"
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    # rescaled_times = times * params['growth_rate']
    # for k in range(1,k_max+1):
    #     label = "l=" + str(k)
    #     ax.plot(rescaled_times, ltt_probs[k,:], linewidth=2.0, color=cmap[k-1], label=label)
    #     ax.plot(rescaled_times, sim_ltt_probs[k,:], '--',linewidth=2.0, color=cmap[k-1])
    # ax.set_xlabel('Time',fontsize=12)
    # ax.set_ylabel('q(l,t)',fontsize=12)
    # ax.legend()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.tight_layout()
    # fig.savefig('exp_growth_mdp_ltt_probs_k=4.png', dpi=200,bbox_inches='tight')
    
    
    "Plot simulated p_k probs by k with N(t) on twinned y-axis"
    N_t = params['Ne'] * np.exp(-params['growth_rate']*times)
    rescaled_times = times * params['growth_rate']
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))
    ax2 = ax1.twinx()
    ax2.plot(rescaled_times, N_t, linewidth=2.0, color='k', label='N(t)')
    for k in range(1,k_max+1):
        label = "l=" + str(k)
        ax1.plot(rescaled_times, ltt_probs[k,:], linewidth=2.0, color=cmap[k-1], label=label)
        ax1.plot(rescaled_times, sim_ltt_probs[k,:], '--',linewidth=2.0, color=cmap[k-1])
    ax1.set_xlabel('Time',fontsize=12)
    ax1.set_ylabel('q(l,t)',fontsize=12)
    ax2.set_ylabel('N(t)',fontsize=12)
    ax1.legend()
    sns.move_legend(ax1, "upper right", bbox_to_anchor=(1.0, .8), frameon=False) # bbox coords: (x,y)
    ax2.legend()
    sns.move_legend(ax2, "upper right", bbox_to_anchor=(1.0, .9), frameon=False) # bbox coords: (x,y)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('exp_growth_mdp_ltt_probs_k=4.png', dpi=200,bbox_inches='tight')
    

def compute_ltt_probs(times,params,sample_config):
    
    """
        Analytically compute LTT (q_k) probs
        Note: this version allows for sequential sampling through time
    """

    k_max = params['k_max']
    n_times = len(times)
    N_t = params['Ne'] * np.exp(-params['growth_rate']*times)
    
    sample_times, sample_sizes = sample_config
    
    "Pre-compute pairs matrix"
    pairs = np.zeros((k_max+1,k_max+1),int) # matrix of pre-computed binomial coefficients i.e. (k choose 2)
    k_vals = np.arange(0,k_max+1)
    pairs_vec = (k_vals*(k_vals-1))/2
    np.fill_diagonal(pairs, -pairs_vec)
    for k in range(k_max):
        pairs[k,k+1] = (k+1) * (k) / 2
    
    "Numerically solve for p_k probs back through time"
    pk = np.zeros(k_max+1)
    #pk[sample_sizes[0]] = 1.0
    pk[0] = 1.0
    ltt_probs = np.zeros((k_max+1,n_times))
    ltt_probs[:,0] = pk
    for i in range(n_times-1):
        
        dt = times[i+1] - times[i]
        
        # Update pk probs at sampling event
        if times[i] in sample_times: #and i != 0:
            count = int(sample_sizes[sample_times.index(times[i])])
            new_pk = np.zeros(k_max+1)
            for k in range(k_max + 1 - count):
                new_pk[k+count] = pk[k]
            pk = new_pk
        
        # Update based on current coalescent rates
        N = N_t[i]
        _lambda = pairs / N # rates of coalescence
        pk = pk + (np.matmul(_lambda,pk) * dt)
        
        # Ensure positivity and normalize
        pk[pk<0] = 0
        pk = pk / np.sum(pk)
        ltt_probs[:,i+1] = pk
        
    return ltt_probs

def test_k_coal_times(params):
    
    """
        Compare distribution of times for the k-th sampled lineage coalescing with 
        the other k-1 sampled lineage for an exp growing haploid pop
    """

    k_max = 5
    sample_size = k_max
    times = params['times']
    params['samples'] = sample_size
    params['k_max'] = k_max
    
    "Compute LTT p_k probs for coal time pdf"
    N_t = params['Ne'] * np.exp(-params['growth_rate']*times)
    sample_times = [0]
    sample_sizes = [sample_size-1] # minus one because we're not including k-th lineage
    sample_config = (sample_times,sample_sizes)
    p_k = compute_ltt_probs(times,params,sample_config)
    p_k_init = np.zeros(k_max+1)
    p_k_init[sample_sizes[0]] = 1.0
    p_k[:,0] = p_k_init
    
    "Find coalescent time pdf"
    n_times = len(times)
    pdf = np.zeros(n_times)
    k_vals = np.arange(0,k_max+1)
    pdf[0] = np.nan #np.dot(p_k[:,0],k_vals) / N_t[0]
    prob_no_coal = 1.0
    for i in range(n_times-1):
        dt = times[i+1] - times[i]
        _lambda = np.dot(p_k[:,i],k_vals) / N_t[i] # k is the # of other lineages in the tree
        prob_no_coal *= np.exp(-_lambda*dt)
        pdf[i+1] = _lambda * prob_no_coal
    
    "Sim time at which k-th sample coalesces to other sampled lineages"
    sims = 1000
    coal_times = []
    for s in range(sims):
        ts = sim_exp_growth_tree(params,plot=False)
        tree = ts.first()
        rand_sample = randrange(k_max) # randomly sample one tip
        parent_time = tree.time(tree.parent(rand_sample))
        coal_times.append(parent_time)
        
    fig, ax = plt.subplots(figsize=(5,3.5))
    #rescaled_coal_times = np.array(coal_times) * params['growth_rate']
    #rescaled_times = times * params['growth_rate']
    sns.histplot(x=coal_times, stat='density', kde=False, binwidth=2.5,label='Simulated')
    sns.lineplot(x=times, y=pdf, ax=ax, color='black',label='Analytical')
    ax.set_xlabel('Coalescent time',fontsize=12)
    ax.set_ylabel('p(k,t)',fontsize=12)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('exp_growth_mdp_coal_density_k=5.png', dpi=200, bbox_inches='tight')
    

def test_expected_values_single_times(params):
    
    """
        Test expected values of different sampling actions at single time points assuming no other sampling
    """
    
    sample_time_indexes = np.arange(0,76,2)
    sample_times = list(np.flip(times[sample_time_indexes])) # going from past to present with time measured in dist from present
    #action_space = [0.33,0.66,1.0]
    action_space = [2,3,4,5]
    #st_action_counts = get_state_action_counts(params,action_space,sample_times) # sample counts for each state, action pair
    
    s_times = []
    s_actions = []
    #sample_freqs = []
    expected_values = []
    methods = []
    for st, time in enumerate(sample_times):
        print("Time = ",str(time))
        for a in range(len(action_space)):
            action = action_space[a] #st_action_counts[st][a] # action as a sample count
            sample_actions = np.zeros(len(sample_times))
            sample_actions[st] = action
            policy = (sample_times,sample_actions)
            V = expected_value(st,action,policy,params)
            mean_sim_reward = monte_carlo_value(params,policy,sims=100,mle=False)
            
            # Add sim values
            s_times.append(time)
            s_actions.append(action)
            #sample_freqs.append(action_space[a])
            expected_values.append(mean_sim_reward)
            methods.append('Simulated')
            
            # Add expected vales
            s_times.append(time)
            s_actions.append(action)
            #sample_freqs.append(action_space[a])
            expected_values.append(V)
            methods.append('Analytical')
    
    s_times = np.array(s_times) * growth_rate
    dd = {'time': s_times,
          'Sample size': s_actions,
          #'sample_freq': sample_freqs,
          'expected_value': expected_values,
          'Method': methods}
    df = pd.DataFrame.from_dict(dd)
    
    df.to_csv('exp_coal_mdp_single_actions_rewards.csv',index=False)

def enumerate_policy_space(action_space,sample_times):
    
    """
        Enumerate all possible policies in policy space
        by finding all possilble permuations of all actions at all times
    """
    
    action_lists = [action_space for x in range(len(sample_times))] # a list of lists for all actions at all times
    #print ("The original lists are : " + str(action_lists))    
    policy_space = list(itertools.product(*action_lists)) # compute all possible permutations
    #print ("All possible permutations are : " + str(policy_space))
    print("The size of policy space is: " + str(len(policy_space)))
    
    return policy_space

def brute_force_search(params,action_space,sample_times):
    
    """
        Perform a brute force search over all possible policies to find optimal stategy 
        based on Monte Carlo simulation of the MDP
    """
    
    N_st = params['Ne'] * np.exp(-params['growth_rate']*np.array(sample_times)) # N at sample times
    costs_by_time =  params['costs_by_time']
    
    #st_action_counts = get_state_action_counts(params,action_space,sample_times)

    # Enumerate all possible policies in policy space    
    policy_space = enumerate_policy_space(action_space,sample_times)
    policy_num = len(policy_space)
    policy_space = np.array(policy_space) # as np array
    
    # Evaluate expected rewards based on Monte Carlo sims 
    expected_rewards = []
    sim_rewards = []
    costs = []
    mle_errors = []
    for p in range(policy_num):
        sample_fractions = policy_space[p,:]
        sample_actions = np.floor(sample_fractions * N_st) # actions as sample counts
        policy = (sample_times, sample_actions)
        
        # Get expected reward from MDP
        V = eval_policy(policy,params)
        expected_rewards.append(np.sum(V))
        #print('Final policy evaluation:', np.sum(V))
        
        # Run Monto Carlo sims
        sample_count = np.sum(costs_by_time * sample_actions)
        mean_sim_reward, mean_mle_error = monte_carlo_value(params,policy,sims=100,mle=True)
        
        sim_rewards.append(mean_sim_reward)
        costs.append(sample_count)
        print('Policy num =', str(p), ' Expected reward =', f'{V:0.4f}', ' Mean sim reward =', f'{mean_sim_reward:0.4f}')
        
        mle_errors.append(mean_mle_error)
        #print('Policy num =', str(p), ' Mean sim reward =', f'{mean_sim_reward:0.4f}', ' Mean MLE error =', f'{mean_mle_error:0.4f}')
    
    dd = {'action_t'+str(k):policy_space[:,k] for k in range(len(sample_times))}
    dd['expected_reward'] = expected_rewards
    dd['sim_reward'] = sim_rewards
    dd['cost'] = costs
    #dd['mle_error'] = mle_errors
    df = pd.DataFrame.from_dict(dd)
    
    df.to_csv('exp_coal_mdp_rewards_larger_policy_space_constant_costs.csv',index=False)
    
    df['relative_reward'] = df['expected_reward'] / df['expected_reward'].max()
    df['relative_cost'] = df['cost'] / df['cost'].max()
    df['net_reward'] = df['relative_reward'] - df['relative_cost']
    
    df.sort_values('net_reward',ascending=False,inplace=True)
    print('Optimal policy:')
    print(df.iloc[0])


def monte_carlo_value(params,policy,sims=100,mle=True):
    
    """
        Perform Monte Carlo simulations to compute expected value of a sampling policy
        if mle=True: computes mean error in MLE estimate 
    """

    _,sample_sizes = policy
    total_samples = np.sum(sample_sizes)
    sim_rewards = np.zeros(sims)
    mle_errors =  np.zeros(sims)
    if total_samples > 0:
        for s in range(sims):
            
            ts = sim_exp_growth_tree(params,sample_config=policy,plot=False)
            tree = ts.first()
            
            if mle:
                tree_df = expcoal.build_tree_df_from_ts(ts)
                mle =  expcoal.opt_MLE(tree_df,params,bounds=params['mle_bounds'])
                #print("MLE: ", mle)
                mle_errors[s] = np.abs(mle - params['growth_rate'])
            
            coal_times = np.array([tree.time(u) for u in tree.nodes() if tree.is_internal(u)])
            coal_pop_sizes = params['Ne'] * np.exp(-params['growth_rate'] * coal_times)
            FI = np.sum(coal_pop_sizes**-2) #FI = s * Ne**-2
            sim_rewards[s] = FI
    
    if mle:
        return np.mean(sim_rewards), np.mean(mle_errors)
    else:
        return np.mean(sim_rewards)

def get_state_action_counts(params,action_space,sample_times):
    
    """
        Get list of actions (sample counts) for each state
    """

    N_st = params['Ne'] * np.exp(-params['growth_rate'] * np.array(sample_times)) # N at sampling times
    st_action_counts = [] # actions as sample counts from each st
    for st in range(len(sample_times)):
        counts = np.floor(np.array(action_space) * N_st[st])
        st_action_counts.append(list(np.array(counts,int)))
        
    return st_action_counts

def qvals_to_df(V_action):
    
    """
        Convert q-values to pandas df for export to csv file
        Note: Input should be q-values under optimal policy in V_action
    """
    
    s_times = []
    s_actions = []
    expected_values = []
    for st, time in enumerate(sample_times):
        for a in range(len(action_space)):
            action = action_space[a] # action as sample freq
            s_times.append(time)
            s_actions.append(action)
            expected_values.append(V_action[a,st])
    s_times = np.array(s_times) * growth_rate
    dd = {'time': s_times,
          'action': s_actions,
          'expected_qvalue': expected_values}
    df = pd.DataFrame.from_dict(dd)
    df.to_csv('exp_coal_mdp_optimal_policy_q-values.csv',index=False)

def eval_policy(policy,params):
    
    """
        Direcly evaluate expected value of policy over all sampling times
    """
    
    sample_times, sample_actions = policy
    reward = 0
    for next_st in range(len(sample_times)):
        action = int(sample_actions[next_st])
        if action > 0:
            reward += q_value(next_st,action,policy,params)
        
    return reward

def q_value(st,action,policy,params,future_sampling=False):
    
    """
        Compute expected q-value of playing action a from state st under policy
    """
    
    # Get sampling action
    sample_times, sample_actions = policy
    N_t = params['Ne'] * np.exp(-params['growth_rate'] * params['times'])
    
    n_times = len(params['times'])
    k_max = params['k_max']
    
    st_time = sample_times[st]
    st_time_index = np.where(times == st_time)[0][0]
    
    "Compute expected FI reward for each newly added sample based on the expected coal time"
    reward = 0
    for c in range(action): # for each sample
        
        "Compute LTT p_k probs for coal time pdf"
        sample_config_sizes = np.copy(sample_actions)
        if not future_sampling:
            sample_config_sizes[st:] = 0 # set sample sizes at future times to zero
        sample_config_sizes[st] = c # set sample size at current time 
        sample_config = (sample_times,sample_config_sizes)
        p_k = compute_ltt_probs(times,params,sample_config)

        "Find coalescent time pdf if we've sampled at least two lineages"
        pdf = np.zeros(n_times)
        if np.sum(sample_config_sizes) > 0: # had >= two, but I think this should be > 0
            k_vals = np.arange(0,k_max+1)
            pdf[0] = np.dot(p_k[:,0],k_vals) / N_t[0]
            prob_no_coal = 1.0
            #for i in range(n_times-1): # this is what we had before but I think this only worked because we were assuming to future sampling
            for i in range(st_time_index,n_times-1): # going backwards in time from st_time to one past final time in past
                dt = times[i+1] - times[i]
                _lambda = np.dot(p_k[:,i],k_vals) / N_t[i] # k is the # of other lineages in the tree
                prob_no_coal *= np.exp(-_lambda*dt)
                pdf[i+1] = _lambda * prob_no_coal        
            pdf = pdf / np.sum(pdf)
        
        FI = np.dot(pdf,N_t**-2) # expected Fisher Information marginalizing over coal times
        reward += FI
        
    return reward

def iter_policy_eval(policy,params,theta=0.01):
    
    """
        Iteratively evaluate expected value V[s] of policy for each state s
        Algorithim adapted from Barto and Sutton (see pg. 75)
            theta (float): threshold value for estimation accuracy
    """
    
    sample_times, sample_actions = policy
    
    # Evaluate value of policy iteratively
    V = np.ones(len(sample_times)+1)*0.1 # values of sampling actions at each state/time
    V[-1] = 0 # value of terminal state is always zero
    delta = np.inf
    iteration = 0
    while delta > theta:
        delta = 0.
        for st in range(len(sample_times)): # st = state/time index
            v = V[st] # current estimated value
            action = int(sample_actions[st]) # actions are sample counts
            V[st] = q_value(st,action,policy,params) # + V[st+1]
            delta = np.max([delta, np.abs(v-V[st])]) 
        iteration += 1
        print('Iteration =', str(iteration), ' delta =', f'{delta:0.4f}')
        
    return V

def opt_policy(params,action_space,sample_times):
    
    """
        Optimize sampling policy by value iteration.
    """
    
    V = np.ones(len(sample_times)+1)*0.1 # initial values of sampling actions at each state/time
    V[-1] = 0 # value of terminal state is always zero
    V_action = np.zeros(len(action_space)) # expected q-values based on action
    theta = 0.1 # threshold value for estimation accuracy
    delta = np.inf
    iteration = 0
    opt_actions = np.zeros(len(sample_times),dtype=int) # could also be random values
    st_action_counts = get_state_action_counts(params,action_space,sample_times) # sample counts for each state, action pair
    while delta > theta:
        delta = 0.
        for st in range(len(sample_times)): # st = state/time index
            
            v = V[st] # current estimated value
            
            # Update policy to reflect optimal action at earlier times
            sample_actions = [st_action_counts[alt_st][opt_actions[alt_st]] for alt_st in range(len(sample_times))]
            policy = (sample_times,sample_actions)
            
            # Compute value of each action at current state/time
            for a in range(len(action_space)):
                action = st_action_counts[st][a] # action as a sample count
                reward = q_value(st,action,policy,params,future_sampling=True) # + V[st+1]
                reward = reward / max_reward
                cost = costs_by_time[st] * action / max_cost
                V_action[a] = reward - cost

            V[st] = np.max(V_action)
            opt_actions[st] = np.argmax(V_action)
            delta = np.max([delta, np.abs(v-V[st])])
            
        iteration += 1
        print('Iteration =', str(iteration), ' delta =', f'{delta:0.4f}')
    
    print('Final policy evaluation:', np.sum(V))
    opt_actions = [action_space[a] for a in opt_actions] # map actions back to sample freqs
    print('Final policy:', opt_actions)

if __name__ == '__main__':
    

    """
        Define MDP params
        Note: growth_rate should not be too high otherwise msprime sims will differ
        from analytical coalescent densities due to time step discretization effects
        Expected and sim values seem to agree more when growth_rate is low
    """
    Ne = 50.0  # effective pop sizes
    k_max = 50 # max possible # of lineages when computing p_k probs
    growth_rate = 0.1 # intrinsic growth rate
    t_origin = np.abs(np.log(1 / Ne)) / growth_rate # time at which N = 1
    times = np.linspace(0, t_origin, 100) # had 40
    params = {'Ne': Ne,
               'growth_rate': growth_rate,
               'k_max': k_max,
               'times': times}

    params['mle_bounds'] = (0.0,0.3) # only used if estimating mle values

    """
        Test pairwise distribution of coalescent times under exp growth
    """
    # test_pair_coal_times(params)
    
    """
        Test lineage through time (LTT) q_k probs with or without sequential sampling
    """
    # test_ltt_probs(params)
    
    """
        Test distribution of coal times for newly added sample with under exp growth - generalize to sequential sampling
    """
    # test_k_coal_times(params)
    
    """
        Test expected values of different sampling actions at single time points assuming no other sampling
    """
    # test_expected_values_single_times(params)

    
    """
        Run iterative policy evaluation:
            This is not particularly interesting in the case of a deterministic model and policy
            because we could just evaluate the policy directly given the deterministic sequence of states/actions (using eval_policy)
            But for completeness checking for convergence to right expected value
    """
    
    # Define sampling policy to evaluate
    sample_time_indexes = np.arange(0,30,10)
    sample_times = list(np.flip(times[sample_time_indexes])) # going from past to present with time measured in dist from present
    sample_fractions = [0.5,0.5,0.5] # sampling actions are sample fractions
    N_t = params['Ne'] * np.exp(-params['growth_rate']*times)
    sample_actions = np.floor(sample_fractions * N_t[sample_time_indexes]) # actions as sample counts
    
    policy = (sample_times,sample_actions)
    
    V = iter_policy_eval(policy,params,theta=0.01)
    print('Iter policy evaluation:', np.sum(V))
    
    eV = eval_policy(policy,params)
    print('Direct policy evaluation:', eV)
    
    mean_sim_reward = monte_carlo_value(params,policy,sims=100,mle=False)
    print('Mean sim reward:', mean_sim_reward)
    
    """
        Test value iteration against brute_force_search to find optimal policy 
    """
    #sample_time_indexes = np.arange(0,61,12) # for 6 time points (used in paper)
    sample_time_indexes = np.arange(0,61,20) # for 4 time points (reduced space for testing)
    sample_times = list(np.flip(times[sample_time_indexes])) # going from past to present with time measured in dist from present
    #action_space = [0.0,0.25,0.5,0.75,1.0] # for five sampling fractions (used in paper)
    action_space = [0.0,0.33,0.66,1.0] # reduced action space for testing
    st_action_counts = get_state_action_counts(params,action_space,sample_times) # sample counts for each state, action pair
    max_sample_count = np.sum(np.array(st_action_counts)[:,-1])
    params['k_max'] = max_sample_count
    
    """
        Compute time-dependent costs: Here we will assume the cost of sampling
        at the first sampling time is one, but costs decrease lineary through time
        and are X-fold more at the first sampling time
        Set fold_increase = 1 if don't want sampling cost to increase
    """
    fold_increase = 1 # set to one if don't want sampling costs to change over time
    sampling_duration = sample_times[0] - sample_times[-1]
    costs_by_time = [1 + (fold_increase-1) * st / sampling_duration for st in sample_times]
    params['costs_by_time'] = costs_by_time
    
    #Find max reward/costs
    action = action_space[-1] # largest sample size possible
    sample_actions = [st_action_counts[alt_st][-1] for alt_st in range(len(sample_times))]
    policy = (sample_times,sample_actions)
    V = eval_policy(policy,params)
    max_reward = V
    max_cost =  np.sum(costs_by_time * np.array(st_action_counts)[:,-1]) # max_sample_count
    print("Max reward =", f'{max_reward:0.4f}', ' Max cost =', f'{max_cost:0.4f}')

    """
        Optimize policy using value iter algorithm
    """
    opt_policy(params,action_space,sample_times)
    
    """
        Find optimal policy based on brute force search over all possible policies
    """
    brute_force_search(params,action_space,sample_times)
    
    
    

    

    

    

    
    
    
    
    
    