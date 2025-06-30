#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:50:47 2025

The transmission tree MDP module for minimizing transmission distances between infected hosts
in a transmission tree. 

The main functions include:
    iter_policy_eval(): to compute the long-term expected reward or value of a given policy by iterative policy evalution.
    opt_policy(): to optimize sampling policies using by value iteration.
    q_value(): to compute expected value (q-value) of a given state-action pair under a policy
    brute_force_search(): to perform a brute force search over policy space to find optimal policy.

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import stirling2
from scipy.special import binom
import pandas as pd
import itertools

def sim_tree(params,sample_config=None,plot=False):
    
    """
        Simulate transmission tree under forward-time discrete-time model
    """
    
    N = params['N']
    times = params['times']
    
    "Seed initial population"
    curr_nodes = list(range(N))
    node_ids = list(range(N))
    node_times = [times[-1]]*N
    node_parents = [-1]*N
    node_children = [[] for _ in range(N)]
    
    "Simulate transmission tree"
    node_cnt = N-1 # counter for node ids
    for t in times[-2::-1]: # starting from second to last time
        parents = np.random.choice(N, N) # sampling with replacement
        new_nodes = []
        for i in range(N):
            node_cnt += 1
            node_ids.append(node_cnt)
            node_times.append(t)
            p = curr_nodes[parents[i]]
            node_parents.append(p) 
            node_children[p].append(node_cnt)
            node_children.append([])
            new_nodes.append(node_cnt)
        curr_nodes = new_nodes 
        
    """
        Sample nodes based on sample_config
        Important note: even though we are sampling without replacements it is
        possible that we may sample a lineage that has already been sampled at
        a future time point. This is by design since we want to allow for the
        possibility of sampling someone's direct parent.
    """
    sample_times, sample_sizes = sample_config
    samples = []
    for st, time in enumerate(sample_times):
        nodes_at_time = np.where(np.array(node_times) == time)[0]
        drawl = np.random.choice(nodes_at_time, sample_sizes[st], replace=False) # sampling without replacement
        samples.extend(list(drawl))
        
    "Find all nodes ancestral to each sample"
    ancestors = []
    for s in samples:
        ancestors.append(s)
        parent = node_parents[s]
        while parent >= 0:
            ancestors.append(parent)
            parent = node_parents[parent]
        ancestors = list(set(ancestors))
    ancestors.sort()
    
    "Simplify lists to only include nodes ancestral to the sample"
    node_ids = [node_ids[a] for a in ancestors]
    node_times = [node_times[a] for a in ancestors]
    node_parents = [node_parents[a] for a in ancestors]
    node_children = [node_children[a] for a in ancestors] # will still contain children who are not themselves ancestors!
    
    """
        Compute transmission dist matrix between samples:
            Distance here is defined as the number of transmission events/edges linking two individuals
            in the tranmission tree. A direct transmission pair will therefore have a distance
            of one. Number of missing links will therefore be D - 1.
            Lineages that don't coalesce before final time point in past are assumed to
            automatically coalesce in the next generation, which adds one to each lineages
            transmission distance. E.g. Two indvs sampled at the final time will hava a D = 2.
    """
    D = np.zeros((len(samples),len(samples)))
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if j > i: # only compute upper triangle of dist matrix
                ancestors_i = [sample_i]
                parent = node_parents[node_ids.index(sample_i)]
                while parent >= 0:
                    ancestors_i.append(parent)
                    parent = node_parents[node_ids.index(parent)]
                    
                ancestors_j = [sample_j]
                parent = node_parents[node_ids.index(sample_j)]
                while parent >= 0:
                    ancestors_j.append(parent)
                    parent = node_parents[node_ids.index(parent)]
                    
                links_i = set(ancestors_i).difference(set(ancestors_j)) # nodes linking i to mrca
                links_j = set(ancestors_j).difference(set(ancestors_i)) # nodes linking j to mrca
                dist = len(links_i) + len(links_j)
                D[i,j] = dist
    
    D = D + D.T # to get symmetric D matrix from upper triangle of D
    
    "Find everyone's distance to nearest sample"
    node_dists = [] # node dists to nearest sample
    samples_set = set(samples)
    for nd in node_ids:
        dist = 0
        children = set([nd])
        overlap = samples_set.intersection(children)
        while len(overlap) == 0:
            dist += 1
            gc = []  # all children of current children
            for c in children:
                if c in ancestors:
                    gc.extend(node_children[node_ids.index(c)])
            children = set(gc)
            overlap = samples_set.intersection(children)
        node_dists.append(dist)
    
    tree = {'ids': np.array(node_ids),
            'times': np.array(node_times),
            'parents': np.array(node_parents),
            'children': node_children,
            'dists': np.array(node_dists),
            'd_matrix': D}
    
    return tree

def compute_ltt_mk_probs(times,params,sample_config):
    
    """
        Analytically compute LTT (q_k) and missing link (m_k) probs
    """
    
    k_max = params['k_max']
    n_times = len(times)
    N = params['N']
    sample_times, sample_sizes = sample_config
    
    """
        Compute transition probs for i lineages descending from j ancestors in the previous generation
        Formula is from Wakeley book (pg. 62)
    """
    G = np.zeros((k_max+1,k_max+1)) # transition probs
    for i in range(k_max+1):
        for j in range(k_max+1):
            N_j = np.prod(range(N,N-j,-1)) # N * (N -1) * .... (N-j+1), but no +1 because of python indexing
            S_ij = stirling2(i, j) # number of ways i elements can be partitioned into j (non-empty) subsets
            G[i,j] = N_j * S_ij / N**i
            
    "Numerically solve for LTT q_k (p_k) probs back through time"
    pk = np.zeros(k_max+1)
    pk[0] = 1.0
    ltt_probs = np.zeros((k_max+1,n_times))
    
    mk = np.zeros(n_times+1)
    mk[0] = 1.0
    mk_probs = np.zeros((n_times+1,n_times))
    
    for i in range(n_times):

        # Update pk probs at sampling event
        if times[i] in sample_times:
        
            count = int(sample_sizes[sample_times.index(times[i])])
            new_pk = np.zeros(k_max+1)
            f0 = 0 # expected fraction of "new" samples
            for k in range(k_max + 1 - count):
                prob_anc = k / N # prob sample is ancestor assuming we have k sampled lineages
                prob_no_anc = 1 - k / N # prob sample is not an ancestor
                for l in range(count+1):
                    prob_l_new = binom(count,l) * prob_no_anc**l * prob_anc**(count-l)
                    new_pk[k+l] += pk[k] * prob_l_new
                    if (k+l) > 0:
                        f0 += pk[k] * prob_l_new * (count/(k+l))
            pk = new_pk
            
            # Update m_k probs
            new_mk = np.zeros(n_times+1)
            new_mk[0] = f0
            for k in range(1,n_times+1):
                new_mk[k] = (1-f0) * mk[k]
            mk = new_mk
        
        # Record pk/mk at time
        pk[pk<0] = 0
        pk = pk / np.sum(pk)
        ltt_probs[:,i] = pk
        if np.sum(mk) > 0.0:
            mk[mk<0] = 0
            mk = mk / np.sum(mk)
        mk_probs[:,i] = mk
        
        # Update mk
        new_mk = np.zeros(n_times+1)
        for k in range(n_times):
            new_mk[k+1] = mk[k] # all get shifted up one
        mk = new_mk
        
        # Correct mk for coalescent events reducing distance to nearest neighbor
        k_vals = np.arange(0,k_max+1)
        k_vals = k_vals - 1
        k_vals[0] = 0 
        p_coal = np.dot(pk,(k_vals / N)) # k is the # of other lineages in the tree
        new_mk = np.zeros(n_times+1)
        for k in range(n_times+1):
            pm_in = p_coal * mk[k] * np.sum(mk[k+1:]) / 2
            pm_out =  p_coal * mk[k] * np.sum(mk[:k]) / 2
            new_mk[k] = mk[k] + pm_in - pm_out
        mk = new_mk
        
        # Update pk for next time step
        pk = np.matmul(pk,G)
        
        # Ensure positivity and normalize
        pk[pk<0] = 0
        pk = pk / np.sum(pk)
        if np.sum(mk) > 0.0:
            mk[mk<0] = 0
            mk = mk / np.sum(mk)
          
    return ltt_probs, mk_probs  

def test_ltt_probs(params):
    
    """
        Test lineage through time (LTT) p_k probs with sequential sampling
    """
    
    times = params['times'] #np.linspace(0, 50, n_times+1)
    n_times = len(times)
    k_max = params['k_max']
    
    "Set up sample config"
    sample_times = [0,5] # needs to be a subset of times
    sample_sizes = [2,2]
    sample_config = (sample_times,sample_sizes)
    k_max = 4
    params['k_max'] = k_max
    
    # For sampling every step
    #sample_times = list(times)
    #sample_sizes = [8] * len(times)
    #sample_config = (sample_times,sample_sizes)
    
    "Analytically compute LTT (p_k) probs"
    #ltt_probs = compute_ltt_probs(times,params,sample_config)
    ltt_probs, mk_probs = compute_ltt_mk_probs(times,params,sample_config)
    
    "Simulate distribution of LTT"
    sims = 1000
    sim_ltt = np.zeros((n_times+1,n_times,sims),int) # remove third axis for sims
    for s in range(sims):
        tree = sim_tree(params,sample_config)
        for i in range(n_times):
            anc_nodes = np.where(tree['times'] == times[i])[0] # nodes alive at time i
            anc_node_dists = tree['dists'][anc_nodes]
            b = np.bincount(anc_node_dists, minlength=n_times+1) # count lineages by dist
            sim_ltt[:,i,s] = b
    
    "Compute total LTT probs from sims"    
    total_ltt = np.sum(sim_ltt,axis=0).transpose() # sum over rows/dists to get total number of anc lineages
    sim_ltt_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=k_max+1), axis=0, arr=total_ltt) # apply bincount over times so returned arrays has counts as rows and times as columns
    sim_ltt_probs = sim_ltt_counts / sims
    
    """
        Compute average missing link probs from sims
    """
    sim_link_probs = np.zeros((n_times+1,n_times,sims))
    for s in range(sims):
        row_sums = sim_ltt[:,:,s].sum(axis=0)
        sim_link_probs[:,:,s] = sim_ltt[:,:,s] / row_sums[np.newaxis,:]
    sim_link_probs = np.mean(sim_link_probs,axis=2)
    l_series = np.arange(n_times+1,dtype=int)
    l_matrix = np.tile(l_series,(n_times,1)).transpose()
    sim_avg_mlinks = np.sum(l_matrix * sim_link_probs, axis=0)
    
    "Analytically compute missing link (m_k) probs"
    #mk_probs = compute_mk_probs(times,params,sample_config,ltt_probs)
    avg_mlinks = np.sum(l_matrix * mk_probs, axis=0)
    
    "Plot expected/average number of lineages through time"
    # k_series = np.arange(k_max+1,dtype=int)
    # k_matrix = np.tile(k_series,(n_times,1)).transpose()
    # expt_ltt = np.sum(k_matrix * ltt_probs, axis=0)
    # sim_expt_ltt = np.sum(k_matrix * sim_ltt_probs, axis=0)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    # ax.plot(times, expt_ltt, linewidth=2.0, label='Analytical')
    # ax.plot(times, sim_expt_ltt, '--',linewidth=2.0, label='Simulated')
    # ax.set_xlabel('Time',fontsize=12)
    # ax.set_ylabel('Expected LTT',fontsize=12)
    # ax.legend()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.tight_layout()
    # fig.savefig('mlinks_mdp_expected_ltt_k=4.png', dpi=200,bbox_inches='tight')
    
    "Plot marginal q_k densities by k"
    cmap = sns.color_palette("colorblind", k_max)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for k in range(1,k_max+1):
        label = "l=" + str(k)
        ax.plot(times, ltt_probs[k,:], linewidth=2.0, color=cmap[k-1], label=label)
        ax.plot(times, sim_ltt_probs[k,:], '--',linewidth=2.0, color=cmap[k-1])
    ax.set_xlabel('Time',fontsize=12)
    ax.set_ylabel('q(l,t)',fontsize=12)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('mlinks_mdp_ltt_probs_k=4.png', dpi=200,bbox_inches='tight')
    
    "Plot expected/average m_k missing links densities by k"
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(times, avg_mlinks, linewidth=2.0, color='slategray', label='Analytical')
    ax.plot(times, sim_avg_mlinks, '--',linewidth=2.0, color='cornflowerblue', label='Simulated')
    ax.legend()
    ax.set_xlabel('Time',fontsize=12)
    ax.set_ylabel('Mean distance to sample',fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('mlinks_mdp_mk_probs_k=4.png', dpi=200,bbox_inches='tight')

def test_vk_probs(params):
    
    """
        Test distance to nearest sampeled neightbor probability dist v(k)
    """
    
    st = 0 # present (sample) time
    action = 2
    sample_times = [0,5] # need to be ordered from past to present
    sample_actions = [2,2]
    
    # Get sampling action
    N = params['N']
    n_times = len(params['times'])
    k_max = params['k_max']
    
    "Compute LTT q_k probs for coal time pdf"
    count = action
    sample_config_sizes = np.copy(sample_actions)
    sample_config_sizes[st] = count - 1 # set sample size at current time to k for the k+1th test sample
    sample_config = (sample_times,sample_config_sizes)
    p_k, m_k = compute_ltt_mk_probs(times,params,sample_config)
    
    "Find pdf for missing links if we've sampled at least two lineages"
    pdf = np.zeros(2*n_times + 1)
    st_time = sample_times[st]

    k_vals = np.arange(0,k_max+1)
    for i in range(st_time,n_times+1): # going backwards in time from st_time to one past final time in past
        sample_dist = i - st_time # sample distance to coalescent event
        if i == st_time:
            
            # Coal at time of sampling
            corr_k_vals = k_vals - (count - 1) # I think this should maybe be minus 1 otherwise we're not accounting for the lineage we're adding
            corr_k_vals[corr_k_vals < 0] = 0
            prob_coal_this_gen = np.dot(p_k[:,i],(corr_k_vals / N)) # correction for newly sampled lineages
            
            corr_m_k = np.zeros(n_times+1)
            for j in range(1,n_times+1):
                corr_m_k[j] = m_k[j,i]
            if np.sum(corr_m_k) > 0:
                corr_m_k = corr_m_k / np.sum(corr_m_k)
            else:
                corr_m_k[0] = 1.0

            for j in range(n_times): # should this be n_times + 1
                pdf[j + sample_dist] += prob_coal_this_gen * corr_m_k[j]
        elif (i == n_times):
            # No coal before final time - so coal at n_times + 1
            prob_coal_this_gen = 1.0 # all remaining lineages must coalesce
            for j in range(n_times): # should this be n_times + 1
                pdf[j + sample_dist + 1] += prob_coal_this_gen * m_k[j,-1] #draw dist of other lineage from m_k[:,-1]
        else:
            # Coal between sample time and final time
            prob_coal_this_gen = np.dot(p_k[:,i],(k_vals / N)) # k is the # of other lineages in the tree
            for j in range(n_times): # should this be n_times + 1
                pdf[j + sample_dist] += prob_coal_this_gen * m_k[j,i]
    
    # Compute probs for min dists
    v = np.zeros(2*n_times + 1)
    for k in range(2*n_times + 1):
        v[k] = pdf[k] * np.prod(1-pdf[:k])
    #mdf[-1] = 1 - np.sum(mdf)
    v = v / np.sum(v) # this seems to work better for inverse weights
    
    print("V(k):", v)
    
    sims = 10000
    policy = (sample_times,sample_actions)
    sim_dists = np.zeros(sims,dtype=int) # sum of min distance between samples
    for s in range(sims):
        tree = sim_tree(params,sample_config=policy,plot=False)
        D = tree['d_matrix']
        np.fill_diagonal(D, np.inf) 
        sim_dists[s] = np.min(D[0,:]) # latest sample will have lowest index in D
    
    
    sim_v = np.bincount(sim_dists, minlength=2*n_times+1) # count lineages by dist
    sim_v = sim_v / np.sum(sim_v)
    print("Sim V(k):", sim_v)
    
    fig, ax = plt.subplots(figsize=(5,3.5))

    barWidth = 0.3
    brd1 = list(range(2*n_times + 1)) 
    brd2 = [x + barWidth for x in brd1]  
    cap = 11
    plt.bar(brd1[:cap], v[:cap], color ='slategray', width = barWidth, 
        edgecolor ='grey', label ='Analytical') 
    plt.bar(brd2[:cap], sim_v[:cap], color ='cornflowerblue', width = barWidth, 
        edgecolor ='grey', label ='Simulated') 

    ax.set_xlabel('Distance k',fontsize=12)
    ax.set_ylabel('v(k)',fontsize=12)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('mlinks_mdp_vk_density_k=4.png', dpi=200, bbox_inches='tight')

def test_expected_value_single_time(params):

    """
        Test expected value of sampling at a single time point assuming no other sampling
    """
    
    st = 0 # present (sample) time
    action = 10
    sample_times = [0] # need to be ordered from past to present
    sample_actions = [10]
    
    # Sampling every time step
    #st = 8
    #action = 6
    #sample_times = list(np.flip(times[:-1]))
    #sample_actions = [6] * len(sample_times)
    
    policy = (sample_times,sample_actions)
    V = q_value(st,action,policy,params)
    print('Expected reward = ', f"{V:.2f}")
    #Vh = expected_value_heterochronous(st,action,policy,params)
    #print('Expected reward before = ', f"{Vh:.2f}")
    mean_sim_dist = monte_carlo_value(params,policy,sims=100)
    print('Mean sim reward = ', f"{mean_sim_dist:.2f}")

def test_constant_values(params):
    
    """
        Test expected values by comparing against simulated values of sampling at
        a constant sampling fraction through time.
    """

    sample_times = list(np.flip(times)) # going from past to present with time measured in dist from present
    action_space = list(np.arange(0.0,1.1,0.1)) # list([0.6])
    
    s_actions = []
    s_freqs = []
    expected_values = []
    methods = []
        
    for a_freq in action_space:
       
        print("Sampling freq = ",str(a_freq))
        
        action = int(a_freq * N) # convert sample freq action to number of samples
        sample_actions = np.ones(len(sample_times),dtype=int) * action
        policy = (sample_times,sample_actions)
        
        "Now looping over sampling times (st) here"
        V = 0
        for st, st_time in enumerate(sample_times):
            V += q_value(st,action,policy,params)
        
        mean_sim_reward = monte_carlo_value(params,policy,sims=100)
        
        # Add sim values
        s_actions.append(action)
        s_freqs.append(a_freq)
        expected_values.append(mean_sim_reward)
        methods.append('Simulated')
        
        # Add expected vales
        s_actions.append(action)
        s_freqs.append(a_freq)
        expected_values.append(V)
        methods.append('Analytical')
    
    dd = {'Sample size': s_actions,
          'Sample freq': s_freqs,
          'Expected value': expected_values,
          'Method': methods}
    df = pd.DataFrame.from_dict(dd)
    
    df.to_csv('mlinks_mdp_constant_fractions_direct-weights_N20_rewards.csv',index=False)
    

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

    N = params['N']

    # Enumerate all possible policies in policy space    
    policy_space = enumerate_policy_space(action_space,sample_times)
    policy_num = len(policy_space)
    policy_space = np.array(policy_space) # as np array
    
    # Evaluate expected rewards based on Monte Carlo sims 
    expected_rewards = []
    sim_rewards = []
    costs = []
    for p in range(policy_num):
        
        sample_actions = policy_space[p,:]
        
        # Convert sample fractions to integer counts
        sample_actions = [int(np.floor(N*freq)) for freq in sample_actions]
        
        policy = (sample_times, sample_actions)
        
        # Get expected reward from MDP
        V = eval_policy(policy,params)
        expected_rewards.append(np.sum(V))
        
        # Run Monto Carlo sims
        sample_count = np.sum(sample_actions)
        mean_sim_reward = monte_carlo_value(params,policy,sims=100)
        
        sim_rewards.append(mean_sim_reward)
        costs.append(sample_count)
        
        print('Policy num =', str(p), ' Mean sim reward =', f'{mean_sim_reward:0.4f}')
    
    dd = {'action_t'+str(k):policy_space[:,k] for k in range(len(sample_times))}
    dd['expected_reward'] = expected_rewards
    dd['sim_reward'] = sim_rewards
    dd['cost'] = costs
    df = pd.DataFrame.from_dict(dd)
    
    #df['relative_reward'] = df['sim_reward'] / df['sim_reward'].max()
    #df['relative_cost'] = df['cost'] / df['cost'].max()
    df['net_reward'] = df['sim_reward'] - df['cost']
    
    df.to_csv('mlinks_mdp_policy_rewards_constant_inverse.csv',index=False)
    
    df.sort_values('net_reward',ascending=False,inplace=True)
    print('Optimal policy:')
    print(df.iloc[0])


def monte_carlo_mk(params,policy,sims=100):
    
    """
        Run sims to obtain Monte Carlo approx to missing links mk density 
    """
    
    times = params['times'] #np.linspace(0, 50, n_times+1)
    n_times = len(times)
    
    "Simulate distribution of missing link probs"
    sim_ltt = np.zeros((n_times+1,n_times,sims),int) # remove third axis for sims
    sim_link_probs = np.zeros((n_times+1,n_times,sims))
    for s in range(sims):
        tree = sim_tree(params,sample_config=policy)
        for i in range(n_times):
            anc_nodes = np.where(tree['times'] == times[i])[0] # nodes alive at time i
            anc_node_dists = tree['dists'][anc_nodes]
            b = np.bincount(anc_node_dists, minlength=n_times+1) # count lineages by dist
            sim_ltt[:,i,s] = b
        row_sums = sim_ltt[:,:,s].sum(axis=0)
        sim_link_probs[:,:,s] = sim_ltt[:,:,s] / row_sums[np.newaxis,:]
    
    sim_link_probs = np.mean(sim_link_probs,axis=2)
    
    return sim_link_probs

def monte_carlo_value(params,policy,sims=100):
    
    """
        Perform Monte Carlo simulations to compute expected value of a sampling policy
    """

    _,sample_sizes = policy
    total_samples = np.sum(sample_sizes)
    sim_rewards = np.zeros(sims) # sum of min distance between samples
    if total_samples > 0:
        for s in range(sims):
            tree = sim_tree(params,sample_config=policy,plot=False)
            D = tree['d_matrix']
            np.fill_diagonal(D, np.inf)
            sim_dists = np.min(D,axis=1)
            
            # If using inverse weighting on dists
            if reward_func == 'inverse':
                rewards = 1 / sim_dists # if using inverse weighting
            
            # If only rewarding direct pairs
            if reward_func == 'direct':
                rewards = sim_dists[sim_dists == 1.]
            
            sim_rewards[s] = np.sum(rewards) # sum of min dists
            #print('Sim:', s, 'Min dist:', f"{avg_min_dist:.2f}")
    
    # Plot dist of simulated rewards across sims
    #f, ax = plt.subplots(figsize=(5,3))
    #sns.histplot(x=sim_rewards, stat='count', kde=True, binwidth=0.5)
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Density')
    
    return np.mean(sim_rewards)

def eval_policy(policy,params):
    
    """
        Evaluate expected value of policy over all sampling times
    """
    
    sample_times, sample_actions = policy
    reward = 0
    for next_st in range(len(sample_times)):
        action = sample_actions[next_st]
        if action > 0:
            reward += q_value(next_st,action,policy,params)
        
    return reward

def iter_policy_eval(policy,params,theta=0.01):
    
    """
        Iteratively evaluate expected value V[s] of policy for each state s
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

def q_value(st,action,policy,params):
    
    """
        Compute expected q-value of playing action a from state st under the policy
    """
    
    # Get sampling action
    sample_times, sample_actions = policy
    N = params['N']
    n_times = len(params['times'])
    k_max = params['k_max']
    
    "Compute LTT p_k probs for coal time pdf"
    count = action
    sample_config_sizes = np.copy(sample_actions)
    """
        CAUTION: We might want to uncomment the line below 
        if we want to exclude value of future samples for value iteration
    """
    #sample_config_sizes[st+1:] = 0 # set sample sizes at future times to zero 
    sample_config_sizes[st] = count - 1 # set sample size at current time to k for the k+1th test sample
    sample_config = (sample_times,sample_config_sizes)
    p_k, m_k = compute_ltt_mk_probs(times,params,sample_config)
    
    # If substituting m_k's estimated from Monte Carlo sims
    #m_k = monte_carlo_mk(params,policy,sims=100)
    
    "Find pdf for missing links if we've sampled at least two lineages"
    pdf = np.zeros(2*n_times + 1)
    st_time = sample_times[st]
    reward = 0
    if np.sum(sample_config_sizes) > 0:
        k_vals = np.arange(0,k_max+1)
        for i in range(st_time,n_times+1): # going backwards in time from st_time to one past final time in past
            sample_dist = i - st_time # sample distance to coalescent event
            if i == st_time:
                
                # Coal at time of sampling
                corr_k_vals = k_vals - (count - 1) # I think this should maybe be minus 1 otherwise we're not accounting for the lineage we're adding
                corr_k_vals[corr_k_vals < 0] = 0
                prob_coal_this_gen = np.dot(p_k[:,i],(corr_k_vals / N)) # correction for newly sampled lineages
                #prob_coal_this_gen = np.dot(p_k[:,i],(k_vals / N))
                
                "Correction seems to make things significantly better when implemented correctly!"
                corr_m_k = np.zeros(n_times+1)
                for j in range(1,n_times+1):
                    corr_m_k[j] = m_k[j,i]
                if np.sum(corr_m_k) > 0:
                    corr_m_k = corr_m_k / np.sum(corr_m_k)
                else:
                    corr_m_k[0] = 1.0

                for j in range(n_times): # should this be n_times + 1
                    pdf[j + sample_dist] += prob_coal_this_gen * corr_m_k[j]
            elif (i == n_times):
                # No coal before final time - so coal at n_times + 1
                prob_coal_this_gen = 1.0 # all remaining lineages must coalesce
                for j in range(n_times): # should this be n_times + 1
                    pdf[j + sample_dist + 1] += prob_coal_this_gen * m_k[j,-1] #draw dist of other lineage from m_k[:,-1]
            else:
                # Coal between sample time and final time
                prob_coal_this_gen = np.dot(p_k[:,i],(k_vals / N)) # k is the # of other lineages in the tree
                for j in range(n_times): # should this be n_times + 1
                    pdf[j + sample_dist] += prob_coal_this_gen * m_k[j,i]
        
        # Compute probs for min dists
        mdf = np.zeros(2*n_times + 1)
        for k in range(2*n_times + 1):
            mdf[k] = pdf[k] * np.prod(1-pdf[:k])
        #mdf[-1] = 1 - np.sum(mdf)
        mdf = mdf / np.sum(mdf) # this seems to work better for inverse weights
    
        # If using inverse weighting on dists
        r_vals = np.ones(2*n_times + 1)
        if reward_func == 'inverse':
            d_vals = np.arange(0,2*n_times + 1) # distances in tree
            d_vals[0] = 1 # impossilbe to have a distance of zero but so we don't divide by zero below
            r_vals = 1 / d_vals
        
        # If only rewarding direct pairs
        if reward_func == 'direct':
            r_vals = np.zeros(2*n_times + 1)
            r_vals[1] = 1.0
        
        #reward += sample_actions[st_time] * np.dot(mdf,r_vals)
        reward += count * np.dot(mdf,r_vals)
    
    return reward

def opt_policy(params,action_space,sample_times):
    
    """
        Optimize sampling policy by value iteration - conditioning on optimal actions at previous steps
    """
    
    V = np.ones(len(sample_times)+1)*0.1 # initial values of sampling actions at each state/time
    V[-1] = 0 # value of terminal state is always zero
    V_action = np.zeros(len(action_space)) # expected values based on action
    theta = 0.001 # threshold value for estimation accuracy
    delta = np.inf
    iteration = 0
    opt_actions = np.ones(len(sample_times),dtype=int) * 8 # current best actions
    while delta > theta:
        delta = 0.
        for st in range(len(sample_times)): # st = state/time index
            
            v = V[st] # current estimated value

            # Compute background rewards before adding new samples
            sample_actions = np.copy(opt_actions)
            sample_actions[st] = 0
            policy = (sample_times,sample_actions)
            background_reward = eval_policy(policy,params)
            
            # Compute value of each action at current state/time
            for a in range(len(action_space)):
                
                action = action_space[a]
                
                sample_actions = np.copy(opt_actions)
                sample_actions[st] = action
                policy = (sample_times,sample_actions)
                
                # Update all rewards by sweeping through states
                reward = eval_policy(policy,params)
                
                # Compute rewards/costs
                reward = (reward - background_reward) # / (max_reward - background_reward)  # relative reward
                cost = action # / N # relative cost
                
                V_action[a] = 1.0 * reward - cost
                
            V[st] = np.max(V_action)
            opt_actions[st] = action_space[np.argmax(V_action)]
            delta = np.max([delta, np.abs(v-V[st])])
            
        iteration += 1
        print('Iteration =', str(iteration), ' delta =', f'{delta:0.4f}', ' Policy:', opt_actions)
        
    print('Final policy evaluation:', np.sum(V))
    print('Final policy:', opt_actions)

if __name__ == '__main__':

    """
        Define MDP params
    """
    N = 10  # pop size
    k_max = 20 # max possible # of lineages when computing p_k probs - in theory shouldn't need to be larger than N but we get weird results if not using 2*N
    times = np.arange(0,10,1,dtype=int) # had 5 or 10 times before
    params = {'N': N,
              'k_max': k_max,
              'times': times}
    reward_func = 'inverse' # global param (inverse of direct)
    
    """
        Test lineage through time (LTT) q_k probs with or without sequential sampling
    """
    # test_ltt_probs(params)
    
    """
        Test distance to nearest sampeled neightbor probability dist v(k)
    """
    # test_vk_probs(params)
    
    """
        Test expected value of sampling at a single time point assuming no other sampling
    """
    # test_expected_value_single_time(params)
    
    """
        Test expected values with constant sampling fractions through time.
    """
    # test_constant_values(params)
    
    """
        Run iterative policy evaluation:
    """
    
    # Define sampling policy to evaluate
    sample_times = list(np.flip(times)) # should be in reverse order
    sample_actions = np.ones(len(sample_times),dtype=int) * 4
    
    policy = (sample_times,sample_actions)
    
    V = iter_policy_eval(policy,params,theta=0.01)
    print('Iter policy evaluation:', np.sum(V))
    
    eV = eval_policy(policy,params)
    print('Direct policy evaluation:', eV)
    
    
    """
        Test value iteration against brute_force_search to find optimal policy 
    """
    sample_times = list(np.flip(times)) # should be in reverse order
    action_space = list(np.arange(2,11,2)) # if using counts
    
    #Find max reward/costs
    action = action_space[-1] # largest sample size possible
    sample_actions = np.ones(len(sample_times),dtype=int) * action
    policy = (sample_times,sample_actions)
    max_reward = eval_policy(policy,params)
    max_cost = np.sum(sample_actions) #int(action_space[-1] * N * len(sample_times))
    print("Max reward =", f'{max_reward:0.4f}', ' Max cost =', f'{max_cost:0.4f}')
    
    """
        Optimize policy using value iter algorithm
    """
    opt_policy(params,action_space,sample_times)
    
    """
        Find optimal policy based on brute force search over all possible policies
    """
    sample_times = list(times)
    action_space = [0.0,0.25,0.5,0.75,1.0]
    
    brute_force_search(params,action_space,sample_times)

    



