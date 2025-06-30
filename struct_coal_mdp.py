#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:09:53 2025

Structured coalescent MDP module for optimizing sampling to estimate migrations rates under
the structured coalescent.

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
import random
import struct_coal_like as structcoal
import itertools
import subprocess
import pyslim
import tskit

global simulator 
simulator = 'slim' # options are slim or msprime

def sim_struct_tree(params,sample_config=None,plot=False,keep_unary=False):
    
    """
        Simulate under island (metapopulation) demographic model with possilbe exponential growth
        See here how to sample serially through time:
            https://tskit.dev/msprime/docs/stable/ancestry.html#sampling-time
        Important: Init pop sizes are sizes at present such that size of population is se^{rt} at time t in the past
    """
    
    init_Ne = params['Ne']
    growth_rate = params['growth_rate']
    M = params['M']
    beta = params['beta']
    
    if simulator == 'msprime':
    
        # Using island model
        # demography = msprime.Demography.island_model(init_Ne, M, growth_rate=growth_rate)
        
        # Using custom demographic model
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=init_Ne[0], growth_rate=growth_rate[0])
        demography.add_population(name="B", initial_size=init_Ne[1], growth_rate=growth_rate[1])
        
        # For symmetric migration
        # demography.set_symmetric_migration_rate(["A", "B"], M)
        
        # For assymetric migration
        demography.set_migration_rate(source="p1", dest="p2", rate=params['m12'])
        demography.set_migration_rate(source="p2", dest="p1", rate=params['m21'])
        
        if sample_config:
        
            # Sample at multiple time points
            sample_times, sample_sizes = sample_config
            
            sample_sets = []
            for time, sizes in zip(sample_times, sample_sizes):
                sample_sets.append(msprime.SampleSet(sizes[0], time=time, population=0))
                sample_sets.append(msprime.SampleSet(sizes[1], time=time, population=1))
                    
            ts = msprime.sim_ancestry(samples=sample_sets, ploidy=1, demography=demography, record_migrations=True, additional_nodes=(msprime.NodeType.MIGRANT), coalescing_segments_only=False)
            
        else: 
        
            # Sample at present
            ts = msprime.sim_ancestry(params['samples'], ploidy=1, demography=demography)
            
    if simulator == 'slim':
        
        sim_script = "./WF_assymetric_migration.slim"
        generations = params['t_origin']
        slim_Ne = int(init_Ne[0] / 2) # SLiM assumes diploid pops so divide by 2 to get haploid size
        ne_param_string = "ne=" + str(slim_Ne)
        m12_param_string = "monetwo=" + str(beta[0,1])
        m21_param_string = "mtwoone=" + str(beta[1,0])
        g_param_string = "g=" + str(generations)
    
        subprocess.run(["slim", "-d", ne_param_string, "-d", m12_param_string, "-d", m21_param_string, "-d", g_param_string, sim_script])
        
        tree = tskit.load("./two_pop_migration.trees")
        ts = sample_slim_tree(tree, sample_config, keep_unary=keep_unary)
        tree = ts.first()
        if tree.has_multiple_roots:
            print("WARNING: Simulated tree has multiple roots which may cause discrepencies with coalescent model")
    
    if plot:
        #print(demography)
        print()
        for tree in ts.trees():
            print("-" * 20)
            print("tree {}: interval = {}".format(tree.index, tree.interval))
            print(tree.draw(format="unicode"))
        print(ts.tables.nodes)
        #print(ts.tables.edges)
        print(ts.tables.migrations)
        print()
    
    return ts

def sample_slim_tree(ts, sample_config, keep_unary=False):
    
    """
        Sample individuals in each population at different times
    """
    
    keep_nodes = []
    data = []

    sample_times, sample_actions = sample_config # sample_times need to correspond to generations in backwards time

    for time_idx, g in enumerate(sample_times):

        # find identifiers for individuals at each generation
        inds = pyslim.individuals_alive_at(ts, g)
        subpops = np.array([ts.individual(i).population for i in inds])
        
        num_samples_p1 = sample_actions[time_idx][0]
        inds_p1 = inds[subpops == 1]
        sample_ids_p1 = np.random.choice(inds_p1, num_samples_p1, replace = False) # indices of random sample
        
        num_samples_p2 = sample_actions[time_idx][1]
        inds_p2 = inds[subpops == 2]
        sample_ids_p2 = np.random.choice(inds_p2, num_samples_p2, replace = False) # indices of random sample
        
        sample_ids = np.concatenate([sample_ids_p1, sample_ids_p2])
        
        for samp in sample_ids:
            keep_nodes.append(ts.individual(samp).nodes[0]) #keep only one genome per individual
            
            # Assign information to list to make dataframe
            individual = ts.individual(samp).id
            node = ts.individual(samp).nodes[0]
            location = ts.individual(samp).location
            
            data.append([individual, node, location])

    "Simplify Tree based on sampling"        
    keep_nodes = np.unique(keep_nodes)
    ts = ts.simplify(keep_nodes, keep_input_roots=True, keep_unary=keep_unary)
    
    return ts


def compute_ltt_probs(times,params,sample_config,finite_size_corr=False):
    
    """
        Compute LTT (q_k) probs jointly for both populations
    """

    k_max = params['k_max']
    n_times = len(times)
    beta = params['beta'] # set birth/migration rates
    N_t_pop0 = params['Ne'][0] * np.exp(-params['growth_rate'][0]*times)
    N_t_pop0[N_t_pop0 < 1.0] = 1.0 # for numerical stability
    N_t_pop1 = params['Ne'][1] * np.exp(-params['growth_rate'][1]*times)
    N_t_pop1[N_t_pop1 < 1.0] = 1.0 # for numerical stability
    
    sample_times, sample_sizes = sample_config


    # init joint density pk for number of lineages in both pops    
    pk = np.zeros((k_max+1,k_max+1))
    pk[0,0] = 1.0
    
    ltt_probs = np.zeros((k_max+1,n_times,2)) # for two pops
    pk_pop0_init = np.zeros(k_max+1)
    pk_pop0_init[int(sample_sizes[0][0])] = 1.0
    ltt_probs[:,0,0] = pk_pop0_init
    pk_pop1_init = np.zeros(k_max+1)
    pk_pop1_init[int(sample_sizes[0][1])] = 1.0
    ltt_probs[:,0,1] = pk_pop1_init
    
    for i in range(n_times-1):
        
        dt = times[i+1] - times[i]
        
        # Update pk probs at sampling event
        if times[i] in sample_times: #and i != 0:
            
            count_pop0 = int(sample_sizes[sample_times.index(times[i])][0])
            count_pop1 = int(sample_sizes[sample_times.index(times[i])][1])
            new_pk = np.zeros((k_max+1,k_max+1))
            for k in range(k_max + 1 - count_pop0):
                for l in range(k_max + 1 - count_pop1):
                    new_pk[k+count_pop0,l+count_pop1] = pk[k,l] 
            pk = new_pk
        
        # Update based on current coalescent rates
        delta_pk = np.zeros((k_max+1,k_max+1))
        for k in range(k_max + 1):
            for l in range(k_max + 1):
                
                # Flow in/out due to coalescent events within pops
                if k < k_max:
                    coal_pop0_in = pk[k+1,l] * (k+1)*k * beta[0,0] / (2 * N_t_pop0[i])
                else:
                    coal_pop0_in = 0                
                coal_pop0_out = pk[k,l] * k*(k-1) * beta[0,0] / (2 * N_t_pop0[i])
                
                if l < k_max:
                    coal_pop1_in = pk[k,l+1] * (l+1)*l * beta[1,1] / (2 * N_t_pop1[i])
                else:
                    coal_pop1_in = 0                
                coal_pop1_out = pk[k,l] * l*(l-1) * beta[1,1] / (2 * N_t_pop1[i])
                
                # Flow in/out due to coalescent/migration events between pops
                if k < k_max:
                    coalmig_pop0_in = pk[k+1,l] * (k+1)*l * beta[1,0] / (N_t_pop1[i]) # birth from pop 1->0 in forward time, scale by pop size of the parent pop N_pop1
                else:
                    coalmig_pop0_in = 0
                coalmig_pop0_out =  pk[k,l] * k * l * beta[1,0] / (N_t_pop1[i]) # birth from pop 1->0 in forward time, scale by pop size of the parent pop N_pop1
                
                if l < k_max:
                    coalmig_pop1_in = pk[k,l+1] * k*(l+1) * beta[0,1] / (N_t_pop0[i]) # birth from pop 0->1 in forward time, scale by pop size of the parent pop N_pop0
                else:
                    coalmig_pop1_in = 0
                coalmig_pop1_out =  pk[k,l] * k * l * beta[0,1] / (N_t_pop0[i]) # birth from pop 0->1 in forward time, scale by pop size of the parent pop N_pop0
                
                #mig_out =  pk[k,l] * k * l * 2 * M / (N_t_pop0[i])
                
                # Flow in/out due to migration at unobserved coalescent events between pops
                if finite_size_corr:
                    
                    """
                        With finite pop/sample size correction 
                        This seems to make LTT probs less accurate unless 
                        we include size of parent pop twice (N_par**2) in the denominators
                        such that we're considering the fraction of unsampled lineages in the parent pop
                        Note: in the future we could also make sure the fraction (Nk - k) / Nk never goes negative  
                    """
                    if l < k_max:
                        mig_01_in = pk[k-1,l+1] * (l+1) * (N_t_pop0[i] - (k-1)) * beta[0,1] / (N_t_pop0[i]**2) # mig from pop 0->1 in forward time
                    else:
                        mig_01_in = 0
                    mig_01_out = pk[k,l] * l * (N_t_pop0[i] - k) * beta[0,1] / (N_t_pop0[i]**2) # mig from pop 0->1 in forward time
                    if k < k_max:
                        mig_10_in = pk[k+1,l-1] * (k+1) * (N_t_pop1[i] - (l-1)) * beta[1,0] / (N_t_pop1[i]**2) # mig from pop 1->0 in forward time
                    else:
                        mig_10_in = 0 
                    mig_10_out = pk[k,l] * k * (N_t_pop1[i] - l) * beta[1,0] / (N_t_pop1[i]**2) # mig from pop 1->0 in forward time
                
                else:
                
                    """
                        Without finite pop/sample size correction
                    """
                    if l < k_max:
                        mig_01_in = pk[k-1,l+1] * (l+1) * beta[0,1] / N_t_pop0[i] # mig from pop 0->1 in forward time
                    else:
                        mig_01_in = 0
                    mig_01_out = pk[k,l] * l * beta[0,1] / N_t_pop0[i] # mig from pop 0->1 in forward time
                    if k < k_max:
                        mig_10_in = pk[k+1,l-1] * (k+1) * beta[1,0] / N_t_pop1[i] # mig from pop 1->0 in forward time
                    else:
                        mig_10_in = 0 
                    mig_10_out = pk[k,l] * k * beta[1,0] / N_t_pop1[i] # mig from pop 1->0 in forward time
                
                delta_pk[k,l] += (coal_pop0_in - coal_pop0_out + coal_pop1_in - coal_pop1_out) * dt # updates from coal events within pops
                delta_pk[k,l] += (coalmig_pop0_in - coalmig_pop0_out + coalmig_pop1_in - coalmig_pop1_out) * dt # updates from coal events between pops
                delta_pk[k,l] += (mig_01_in - mig_01_out + mig_10_in - mig_10_out) * dt # updates from migration due to unobserved coal events between pops
        
        pk += delta_pk
        
        # Ensure positivity and normalize
        pk[pk<0] = 0
        pk = pk / np.sum(pk)
        
        pk_pop0 = np.sum(pk,axis=1)
        pk_pop1 = np.sum(pk,axis=0)
        
        ltt_probs[:,i+1,0] = pk_pop0
        ltt_probs[:,i+1,1] = pk_pop1
        
    return ltt_probs

def ltt_from_ts(ts,times):
    
    """
        Get number of lineages through time from TreeSequence ts    
    """
    
    tree = ts.first()
    ltt = [0,0]
    time_list = []
    ltt_list = []
    type_list = []
    pop_list = []

    flags = np.array(ts.tables.nodes.flags,ndmin=1)         

    for u in tree.timeasc():
        
        pop = tree.population(u)
        event_flag = flags[u]        
        if event_flag == 1: # tree.is_leaf(u):
            event_type = 'sample'
            ltt[pop] += 1
        elif event_flag == 0: #tree.is_internal(u):
            event_type = 'coalescent'
            for ch in tree.children(u):
                ch_pop = tree.population(ch)
                ltt[ch_pop] -= 1
            ltt[pop] += 1 # add back parent
        elif event_flag == 524288:
            event_type = 'migration'
            ch = tree.children(u)[0]
            ch_pop = tree.population(ch)
            ltt[ch_pop] -= 1
            ltt[pop] += 1
        else:
            event_type = 'unknown'
            print("Unrecognized type of node")
            
        time_list.append(tree.time(u))
        ltt_list.append(ltt.copy())
        type_list.append(event_type)
        pop_list.append(pop)
    
    # Get LTT corresponding to times in times
    tree_times = np.array(time_list)
    tree_indexes = np.arange(0,len(tree_times))
    ltt = []
    for t in times:
        list_index =  tree_indexes[tree_times <= t][-1] # find last index in tree times before current time
        ltt.append(ltt_list[list_index])
        
    return ltt

def test_ltt_probs(params):
    
    """
        Test lineage through time (LTT) p_k probs with sequential sampling
    """
    
    times = params['times'] #np.linspace(0, 50, n_times+1)
    n_times = len(times)
    
    "Set up sample config"
    sample_times = [50,0] # needs to be a subset of times
    sample_sizes = [[2,1],[2,1]] # need to have pair of sizes for each pop at each time
    sample_config = (sample_times,sample_sizes)
    
    k_max = 8
    params['k_max'] = k_max
    
    "Analytically compute LTT (p_k) probs"
    ltt_probs = compute_ltt_probs(times,params,sample_config,finite_size_corr=False)
    
    "Simulate distribution of LTT"
    sims = 1000
    sim_ltt_pop0 = np.zeros((k_max+1,n_times,sims),int)
    sim_ltt_pop1 = np.zeros((k_max+1,n_times,sims),int)
    for s in range(sims):
        ts = sim_struct_tree(params,sample_config=sample_config,plot=False)
        ltt = ltt_from_ts(ts,times)
        bit_ltt_pop0 = np.zeros((k_max+1,n_times),int)
        bit_ltt_pop1 = np.zeros((k_max+1,n_times),int)
        for i in range(n_times):
            bit_ltt_pop0[ltt[i][0],i] = 1
            bit_ltt_pop1[ltt[i][1],i] = 1
        sim_ltt_pop0[:,:,s] = bit_ltt_pop0
        sim_ltt_pop1[:,:,s] = bit_ltt_pop1
    sim_ltt_probs_pop0 = np.sum(sim_ltt_pop0,axis=2) / sims
    sim_ltt_probs_pop1 = np.sum(sim_ltt_pop1,axis=2) / sims
    
    sns.set_theme(style="white")
    
    "Plot expected/average number of lineages through time"
    k_series = np.arange(k_max+1,dtype=int)
    k_matrix = np.tile(k_series,(n_times,1)).transpose()
    expt_ltt_pop0 = np.sum(k_matrix * ltt_probs[:,:,0], axis=0)
    expt_ltt_pop1 = np.sum(k_matrix * ltt_probs[:,:,1], axis=0)
    sim_expt_ltt_pop0 = np.sum(k_matrix * sim_ltt_probs_pop0, axis=0)
    sim_expt_ltt_pop1 = np.sum(k_matrix * sim_ltt_probs_pop1, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(times, expt_ltt_pop0, color='cornflowerblue', linewidth=2.0, label='Pop 1')
    ax.plot(times, expt_ltt_pop1, color='orange', linewidth=2.0, label='Pop 2')
    ax.plot(times, sim_expt_ltt_pop0, '--', color='cornflowerblue', linewidth=2.0)
    ax.plot(times, sim_expt_ltt_pop1, '--', color='orange', linewidth=2.0)
    ax.set_xlabel('Time',fontsize=12)
    ax.set_ylabel('Expected LTT',fontsize=12)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('sc_mdp_expected_ltt_k=2-2.png', dpi=200,bbox_inches='tight')
    
    "Plot marginal q_k densities by k"
    # cmap = sns.color_palette("colorblind", k_max)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    # for k in range(1,k_max+1):
    #     label = "l=" + str(k)
    #     ax.plot(times, ltt_probs[k,:], linewidth=2.0, color=cmap[k-1], label=label)
    #     ax.plot(times, sim_ltt_probs[k,:], '--',linewidth=2.0, color=cmap[k-1])
    # ax.set_xlabel('Time',fontsize=12)
    # ax.set_ylabel('q(l,t)',fontsize=12)
    # ax.legend()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # fig.tight_layout()
    # fig.savefig('mlinks_mdp_ltt_probs_k=4.png', dpi=200,bbox_inches='tight')


def test_coal_times(params):
    
    """
        Compare dist of times/locations for the k-th sampled lineage coalescing with 
        the other k-1 sampled lineages
    """

    k_max = 8
    times = params['times']
    params['k_max'] = k_max
    
    "Compute LTT p_k probs for coal time pdf"
    sample_times = [0.] # needs to be a subset of times
    sample_sizes = [[4,3]] # -1 in pop we we sample from
    sample_sizes_sim = [[4,4]] # need to have pair of sizes for each pop at each time
    sample_config = (sample_times,sample_sizes)
    sample_config_sim = (sample_times,sample_sizes_sim)
    ltt_probs = compute_ltt_probs(times,params,sample_config)
    
    """
        Find coalescent time/time pdf
        Would need to this for each pop so pdf should no have dims (2,n_times)
    """
    n_times = len(times)
    N_t_pop0 = params['Ne'][0] * np.exp(-params['growth_rate'][0]*times)
    N_t_pop0[N_t_pop0 < 1.0] = 1.0 # for numerical stability
    N_t_pop1 = params['Ne'][1] * np.exp(-params['growth_rate'][1]*times)
    N_t_pop1[N_t_pop1 < 1.0] = 1.0 # for numerical stability
    
    beta = params['beta']
    
    # Set up coalescent time prob density
    pdf = np.zeros((2,n_times)) # now 2D array for two pops
    k_vals = np.arange(0,k_max+1)
    prob_no_coal = 1.0
    
    # Also track props sample is in each pop
    sample_pop = 1 # pop of newly sampled individual
    s_pr = [0.0,0.0] # sample pop probs
    s_pr[sample_pop] =  1.0
    
    for i in range(n_times-1):
         
        dt = times[i+1] - times[i]
        
        # Compute expected number of lineages in each pop Ak
        A0 = np.dot(ltt_probs[:,i,0],k_vals) # expected num lineage in pop0
        A1 = np.dot(ltt_probs[:,i,1],k_vals) # expected num lineage in pop1
        
        # Migration rates due to unobserved coal events with finite size correction
        mig_01 = s_pr[1] * (N_t_pop0[i] - A0) * beta[0,1] / N_t_pop0[i] # mig from 0 -> 1 in forwards time
        mig_10 = s_pr[0] * (N_t_pop1[i] - A1) * beta[1,0] / N_t_pop1[i] # mig from 1 -> 0 in forwards time
        
        # Migration rates due to unobserved coal events without finite size correction
        #mig_01 = s_pr[1] * beta[0,1] / N_t_pop0[i] # mig from 0 -> 1 in forwards time
        #mig_10 = s_pr[0] * beta[1,0] / N_t_pop1[i] # mig from 1 -> 0 in forwards time
    
        # Update probs sampled lineage is in each pop
        s_pr[0] += (mig_01 - mig_10) * dt
        s_pr[1] += (mig_10 - mig_01) * dt
         
        # Coalescent events within each pop
        prob_coal_pop0 = s_pr[0] * A0 * beta[0,0] / N_t_pop0[i] # k is the # of other lineages in the tree
        prob_coal_pop1 = s_pr[1] * A1 * beta[1,1] / N_t_pop1[i] # k is the # of other lineages in the tree
        
        # Coalescent events between pops due to migration
        prob_coalmig_pop0 = s_pr[1] * A0 * beta[0,1] / N_t_pop0[i] # child in pop1 has parent in pop0 
        prob_coalmig_pop1 = s_pr[0] * A1 * beta[1,0] / N_t_pop1[i] # child in pop0 has parent in pop1 
        
        # Need to track prob of coalescing in each pop
        lambda_pop0 = prob_coal_pop0 + prob_coalmig_pop0
        lambda_pop1 = prob_coal_pop1 + prob_coalmig_pop1
        lambda_total = lambda_pop0 + lambda_pop1
        
        prob_no_coal *= np.exp(-lambda_total*dt)
        pdf[0,i+1] = lambda_pop0 * prob_no_coal
        pdf[1,i+1] = lambda_pop1 * prob_no_coal
    
    # Normalize coal pdf
    pdf = pdf / np.sum(pdf)
    prob_pop0 = np.sum(pdf[0,:]) # / (np.sum(pdf[0,:]) + np.sum(pdf[1,:]))
    
    "Sim time/location at which random indv in sample_pop attaches to sample"
    sims = 1000
    sim_coal_times = np.zeros(sims)
    sim_coal_pops = np.zeros(sims,dtype=int)
    for s in range(sims):
        if s % 100 == 0:
            print("Sim num:",str(s))
        ts = sim_struct_tree(params,sample_config=sample_config_sim,plot=False)
        flags = np.array(ts.tables.nodes.flags,ndmin=1)
        tree = ts.first()
        leaves_in_pop = [lf for lf in tree.leaves() if tree.population(lf) == sample_pop]   
        rand_sample = random.choice(leaves_in_pop) # randomly sample one tip in sample pop
        
        # Traverse back through tree until we hit a coalescent event
        parent = tree.parent(rand_sample)
        parent_type = flags[parent]
        while parent_type != 0: # coalescent event
            parent = tree.parent(parent)
            parent_type = flags[parent] 
        
        sim_coal_times[s] = tree.time(parent) # tree.time(tree.parent(rand_sample)) # parent time
        sim_coal_pops[s] =  tree.population(parent) # tree.population(tree.parent(rand_sample)) # parent pop
    
    # Compute and normalize sim densities from times
    sim_coal_times_pop0 = sim_coal_times[sim_coal_pops == 0]
    sim_coal_times_pop1 = sim_coal_times[sim_coal_pops == 1]
    sim_coal_densities_pop0, _ = np.histogram(sim_coal_times_pop0, bins=times[::1], density=False)
    sim_coal_densities_pop1, _ = np.histogram(sim_coal_times_pop1, bins=times[::1], density=False)
    sim_coal_densities = np.vstack((sim_coal_densities_pop0, sim_coal_densities_pop1))
    sim_coal_densities = sim_coal_densities / np.sum(sim_coal_densities) # normalize denities
    sim_prob_pop0 = np.sum(sim_coal_densities[0,:])
    
    print('Analytcal prob coal in pop 0:', f'{prob_pop0:0.4f}')
    print('Sim prob coal in pop 0:', f'{sim_prob_pop0:0.4f}')
        
    fig, ax = plt.subplots(figsize=(5,3.5))
    pdf[:,0] = np.nan
    ax.plot(times, pdf[0,:], color='cornflowerblue', linewidth=2.0, label='Pop 1')
    ax.plot(times, pdf[1,:], color='orange', linewidth=2.0, label='Pop 2')
    sim_coal_densities[:,0] = np.nan
    ax.plot(times[1::1], sim_coal_densities[0,:], '--', color='cornflowerblue', linewidth=2.0)
    ax.plot(times[1::1], sim_coal_densities[1,:], '--', color='orange', linewidth=2.0)
    ax.set_xlim(-5, 80)
    ax.set_xlabel('Coalescent time',fontsize=12)
    ax.set_ylabel('p(t,u)',fontsize=12)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig('sc_mdp_coal_density_k=2-2_v2.png', dpi=200, bbox_inches='tight')

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

def brute_force_search(params,action_space,sample_times,output_file):
    
    """
        Perform a brute force search over all possible policies to find optimal stategy 
        based on Monte Carlo simulation of the MDP
    """
    
    # Enumerate all possible policies in policy space    
    policy_space = enumerate_policy_space(action_space,sample_times)
    policy_num = len(policy_space)
    policy_space = np.array(policy_space) # as np array
    
    # Evaluate expected rewards based on Monte Carlo sims 
    expected_r_00 = []
    expected_r_01 = []
    expected_r_10 = []
    expected_r_11 = []
    #sim_rewards = []
    sim_r_00 = []
    sim_r_01 = []
    sim_r_10 = []
    sim_r_11 = []
    costs = []
    #mle_errors = []
    mle_m12_errors = []
    mle_m21_errors = []
    for p in range(policy_num):
        
        # Convert sample fractions into sample actions (numbers)
        sample_fractions = policy_space[p,:]
        sample_actions = []
        for st in range(len(sample_times)):
            N0 = params['Ne'][0] * np.exp(-params['growth_rate'][0]*st)
            N1 = params['Ne'][1] * np.exp(-params['growth_rate'][1]*st)
            z0 = int(np.floor(sample_fractions[st][0] * N0))
            z1 = int(np.floor(sample_fractions[st][1] * N1))
            sample_actions.append([z0,z1])
            
        policy = (sample_times, sample_actions)
        
        # Get expected reward from MDP
        V = eval_policy(policy,params)
        expected_r_00.append(V[0,0])
        expected_r_01.append(V[0,1])
        expected_r_10.append(V[1,0])
        expected_r_11.append(V[1,1])
        #print('Policy evaluation:', V)
        
        # Run Monto Carlo sims
        sample_count = np.sum(sample_actions)
        mean_sim_reward, mean_m12_error, mean_m21_error = monte_carlo_value(params,policy,sims=100,get_mle=True)
        
        #sim_rewards.append(mean_sim_reward)
        sim_r_00.append(mean_sim_reward[0,0])
        sim_r_01.append(mean_sim_reward[0,1])
        sim_r_10.append(mean_sim_reward[1,0])
        sim_r_11.append(mean_sim_reward[1,1])
        costs.append(sample_count)
        #mle_errors.append(mean_mle_error)
        mle_m12_errors.append(mean_m12_error)
        mle_m21_errors.append(mean_m21_error)
        #print('Policy num =', str(p), ' Mean MLE error =', f'{mean_mle_error:0.4f}')
        print('Policy num =', str(p), ' Mean MLE error =', f'{mean_m12_error:0.4f}', ' / ',  f'{mean_m21_error:0.4f}')
    
    dd_a0 = {'action_t'+str(k)+'_pop0':policy_space[:,k,0] for k in range(len(sample_times))}
    dd_a1 = {'action_t'+str(k)+'_pop1':policy_space[:,k,1] for k in range(len(sample_times))}
    dd = {**dd_a0, **dd_a1}
    dd['expected_r00'] = expected_r_00
    dd['expected_r01'] = expected_r_01
    dd['expected_r10'] = expected_r_10
    dd['expected_r11'] = expected_r_11
    #dd['sim_reward'] = sim_rewards
    dd['sim_r00'] = sim_r_00
    dd['sim_r01'] = sim_r_01
    dd['sim_r10'] = sim_r_10
    dd['sim_r11'] = sim_r_11
    dd['cost'] = costs
    #dd['mle_error'] = mle_errors
    dd['mle_m12_error'] = mle_m12_errors
    dd['mle_m21_error'] = mle_m12_errors
    df = pd.DataFrame.from_dict(dd)
    
    df.to_csv(output_file,index=False)
    
    # df['relative_reward'] = df['sim_reward'] / df['sim_reward'].max()
    # df['relative_cost'] = df['cost'] / df['cost'].max()
    # df['net_reward'] = df['relative_reward'] - df['relative_cost']
    
    # df.sort_values('net_reward',ascending=False,inplace=True)
    # print('Optimal policy:')
    # print(df.iloc[0])

def monte_carlo_value(params,policy,sims=100,get_mle=False):
    
    """
        Perform Monte Carlo simulations to compute expected value of a sampling policy
        if get_mle=True: computes mean error in MLE estimate 
    """

    _,sample_sizes = policy
    total_samples = np.sum(sample_sizes)
    #sim_rewards = np.zeros(sims)
    sim_rewards = np.zeros((2,2,sims)) # total cumulative reward for each event type
    #mle_errors =  np.zeros(sims)
    mle_m12_errors =  np.zeros(sims)
    mle_m21_errors =  np.zeros(sims)
    if total_samples > 0:
        for s in range(sims):
            
            # Keep unary has to be set True here to retain information about migration events
            ts = sim_struct_tree(params,sample_config=policy,plot=False,keep_unary=True)
            tree = ts.first()
            #mle_ts = ts.simplify(keep_input_roots=True) # create simplified ts for inference
            
            # Run optimization to find ML estimate of current sim tree
            if get_mle:
                
                # If just estimating M
                #mle = structcoal.opt_MLE(ts,params,bounds=params['mle_bounds'])
                #mle_errors[s] = mle - params['M']
                
                # If estimating both migration rates
                mle = structcoal.opt_MLE_params(ts,params,bounds=params['mle_bounds'])
                mle_m12_errors[s] = mle[0] - params['m12']
                mle_m21_errors[s] = mle[1] - params['m21']
            
            # Compute rewards in terms of events
            for nd in tree.postorder():
                if tree.is_internal(nd):
                    parent_pop = tree.population(nd)
                    if len(tree.children(nd)) == 2:
                        child_0 = tree.children(nd)[0]
                        child_1 = tree.children(nd)[1]
                        child_pop_0 = tree.population(child_0)
                        child_pop_1 = tree.population(child_1)
                        
                        # Old way assuming there is one transmission-like event at coalescent events
                        # if child_pop_0 == child_pop_1:
                        #     child_pop = child_pop_0
                        # else:
                        #     if child_pop_0 != parent_pop:
                        #         child_pop = child_pop_0
                        #     else:
                        #         child_pop = child_pop_1
                        # sim_rewards[parent_pop,child_pop,s] += 1
                                
                        # Assuming two individuals are born at coalescent events
                        sim_rewards[parent_pop,child_pop_0,s] += 0.5
                        sim_rewards[parent_pop,child_pop_1,s] += 0.5
    
    if get_mle:
        #return np.mean(sim_rewards,2), np.mean(mle_errors)
        return np.mean(sim_rewards,2), np.mean(mle_m12_errors), np.mean(mle_m21_errors)
    else:
        return np.mean(sim_rewards,2)

def eval_policy(policy,params):
    
    """
        Evaluate expected value of policy over all sampling times
        in terms of the expected number of event types
    """
    
    sample_times, sample_actions = policy
    reward = np.zeros((2,2))
    for next_st in range(len(sample_times)):
        action = sample_actions[next_st]
        if np.sum(action) > 0:
            reward += q_value(next_st,action,policy,params)
        
    return reward

def iter_policy_eval(policy,params,theta=0.01):
    
    """
        Iteratively evaluate expected value V[s] of policy for each state s
    """
    
    sample_times, sample_actions = policy
    
    # Evaluate value of policy iteratively
    V = np.zeros((len(sample_times)+1,2,2)) # values of sampling actions at each state/time
    V[-1] = np.zeros((2,2)) # value of terminal state is always zero
    delta = np.inf
    iteration = 0
    while delta > theta:
        delta = 0.
        for st in range(len(sample_times)): # st = state/time index
            v = V[st] # current estimated value
            action = sample_actions[st] # actions are sample counts
            V[st,:,:] = q_value(st,action,policy,params) # + V[st+1]
            delta = np.max([delta, np.sum(np.abs(v-V[st]))]) 
        iteration += 1
        print('Iteration =', str(iteration), ' delta =', f'{delta:0.4f}')
        
    return V

def q_value(st,action,policy,params,future_sampling=False):
    
    """
        Compute expected q-value of playing action a from state st under the policy
        This version does not shuffle order at which individuals are sampled
    """
    
    # Get sampling action
    sample_times, sample_actions = policy

    times = params['times']
    n_times = len(params['times'])
    k_max = params['k_max']
    k_vals = np.arange(0,k_max+1)
    st_time = sample_times[st]
    st_time_index = list(times).index(st_time) # index of sampling time in times array
    
    N_t_pop0 = params['Ne'][0] * np.exp(-params['growth_rate'][0]*times)
    N_t_pop0[N_t_pop0 < 1.0] = 1.0 # for numerical stability
    N_t_pop1 = params['Ne'][1] * np.exp(-params['growth_rate'][1]*times)
    N_t_pop1[N_t_pop1 < 1.0] = 1.0 # for numerical stability
    beta = params['beta']
    
    # Create sampling 'queue' with samples from each pop in random order
    counts = action
    
    reward = np.zeros((2,2)) # total cumulative reward for each event type
    
    # Sample alternately from each pop until no there are no more samples from each pop
    queue = []
    q0 = [0]*counts[0] # queue of pop 0 samples
    q1 = [1]*counts[1] # queue of pop 0 samples        
    while q0 or q1: # while at least one queue is not empty
        if q0:
            queue.append(q0.pop())
        if q1:
            queue.append(q1.pop())
            
    curr_counts = [0,0]
            
    for q in queue: # q is pop of next sample
    
        sample_config_sizes = np.copy(sample_actions)
        if not future_sampling:
            sample_config_sizes[st+1:] = [0,0] # set sample sizes at future times to zero 
        sample_config_sizes[st] = curr_counts # set sample size at current time to k for the k+1th test sample
        sample_config = (sample_times,sample_config_sizes)
        
        # Update curr_counts
        curr_counts[q] += 1
        
        if np.sum(sample_config_sizes) > 0:
        
            ltt_probs = compute_ltt_probs(times,params,sample_config)
            
            # Set up coalescent time prob density
            pdf = np.zeros((2,n_times)) # now 2D array for two pops
            s_pr_ts = np.zeros((2,n_times))
            prob_no_coal = 1.0
            
            # Also track props sample is in each pop
            s_pr = [0.0,0.0] # sample pop probs
            s_pr[q] = 1.0
            s_pr_ts[:,st_time_index] = s_pr
            
            for i in range(st_time_index,n_times-1):
                 
                dt = times[i+1] - times[i]
                
                # Compute expected number of lineages in each pop Ak
                A0 = np.dot(ltt_probs[:,i,0],k_vals) # expected num lineage in pop0
                A1 = np.dot(ltt_probs[:,i,1],k_vals) # expected num lineage in pop1
                
                # Migration rates due to unobserved coal events with finite size correction
                mig_01 = s_pr[1] * (N_t_pop0[i] - A0) * beta[0,1] / N_t_pop0[i] # mig from 0 -> 1 in forwards time
                mig_10 = s_pr[0] * (N_t_pop1[i] - A1) * beta[1,0] / N_t_pop1[i] # mig from 1 -> 0 in forwards time
                
                # Migration rates due to unobserved coal events without finite size correction
                #mig_01 = s_pr[1] * beta[0,1] / N_t_pop0[i] # mig from 0 -> 1 in forwards time
                #mig_10 = s_pr[0] * beta[1,0] / N_t_pop1[i] # mig from 1 -> 0 in forwards time
            
                # Update state probs for sampled lineage
                #s_pr[0] += (mig_01 - mig_10) * dt
                #s_pr[1] += (mig_10 - mig_01) * dt
                 
                # Update state probs for sampled lineage using MASCO correction - maybe improves accuracy of rewards a little bit
                s_pr[0] += (mig_01 - mig_10 - s_pr[0] * ((A0 * beta[0,0] /  N_t_pop0[i]) + (A1 * beta[1,0] /  N_t_pop1[i]))) * dt
                s_pr[1] += (mig_10 - mig_01 - s_pr[1] * ((A1 * beta[1,1] /  N_t_pop1[i]) + (A0 * beta[0,1] /  N_t_pop0[i]))) * dt
                s_pr = s_pr / np.sum(s_pr)
                
                # Coalescent events within each pop
                prob_coal_pop0 = s_pr[0] * A0 * beta[0,0] / N_t_pop0[i] # k is the # of other lineages in the tree
                prob_coal_pop1 = s_pr[1] * A1 * beta[1,1] / N_t_pop1[i] # k is the # of other lineages in the tree
                
                # Coalescent events between pops due to migration
                prob_coalmig_pop0 = s_pr[1] * A0 * beta[0,1] / N_t_pop0[i] # child in pop1 has parent in pop0 
                prob_coalmig_pop1 = s_pr[0] * A1 * beta[1,0] / N_t_pop1[i] # child in pop0 has parent in pop1 
                
                # Need to track prob of coalescing in each pop
                lambda_pop0 = prob_coal_pop0 + prob_coalmig_pop0
                lambda_pop1 = prob_coal_pop1 + prob_coalmig_pop1
                lambda_total = lambda_pop0 + lambda_pop1
                
                # Update rewards
                reward[0,0] += prob_no_coal * prob_coal_pop0
                reward[0,1] += prob_no_coal * prob_coalmig_pop0
                reward[1,0] += prob_no_coal * prob_coalmig_pop1
                reward[1,1] += prob_no_coal * prob_coal_pop1
                
                prob_no_coal *= np.exp(-lambda_total*dt)
                pdf[0,i+1] = lambda_pop0 * prob_no_coal
                pdf[1,i+1] = lambda_pop1 * prob_no_coal
                
                s_pr_ts[:,i+1] = s_pr
    
    return reward

def opt_policy(params,action_space,sample_times):
    
    """
        Optimize sampling policy by value iteration.
    """
    
    V = np.ones(len(sample_times)+1)*0.1 # initial values of sampling actions at each state/time
    V[-1] = 0 # value of terminal state is always zero
    V_action = np.zeros(len(action_space)) # expected values based on action
    theta = 0.1 # threshold value for estimation accuracy
    delta = np.inf
    iteration = 0
    opt_actions = np.zeros(len(sample_times),dtype=int) # could also be random values
    while delta > theta:
        delta = 0.
        for st in range(len(sample_times)): # st = state/time index
            
            v = V[st] # current estimated value
            
            # Update policy to reflect optimal action at earlier times
            sample_actions = [action_space[opt_actions[alt_st]] for alt_st in range(len(sample_times))]
            policy = (sample_times,sample_actions)
            
            # Compute value of each action at current state/time
            for a in range(len(action_space)):
                
                print('Iteration =', str(iteration), ' State =', str(st), ' Action =', str(a))
                
                action = action_space[a] # action as a sample count
                reward = q_value(st,action,policy,params,future_sampling=True) # + V[st+1]
                reward = np.sqrt(reward[0,1]) + np.sqrt(reward[1,0]) # total migration events
                reward = reward / max_reward
                cost = np.sum(action) / max_cost
                V_action[a] = reward - cost

            V[st] = np.max(V_action)
            opt_actions[st] = np.argmax(V_action)
            delta = np.max([delta, np.abs(v-V[st])])
            
        print('Iteration =', str(iteration), ' delta =', f'{delta:0.4f}')
        iteration += 1
    
    print('Final policy evaluation:', np.sum(V))
    opt_actions = [action_space[a] for a in opt_actions] # map actions back to sample freqs
    print('Final policy:', opt_actions)
    

if __name__ == '__main__':
    
    Ne = [50,50]
    growth_rate = [0.,0.] # for constant sized pops
    M = 0.1 # migration rate (defprecated) -> use m12 and m21
    kappa = 1.0 # migration rate ratio m21 / m12
    m12 = M
    m21 = kappa * M
    t_origin = 4*Ne[0] # for constant sized pops
    split_time = None # time subpops split in distance from t_origin
    times = np.arange(0,t_origin,1)
    params = {'Ne': Ne,
               'growth_rate': growth_rate,
               'M': M,
               'm12': m12,
               'm21': m21,
               'times': times,
               't_origin': t_origin,
               'split_time': split_time,
               'sc_approx': 'masco'}
    #params['mle_bounds'] = (0.001,0.5) # if estimating mle values
    params['mle_bounds'] = ((0.001,0.5),(0.001,0.5)) # if estimating both migration rates (m12 and m21)
    
    """
        Convert migration rates into birth rate matrix beta
    """
    #params['beta'] = np.array([[1-M,M],[M,1-M]]) # for symmetric migration
    params['beta'] = np.array([[1-m21,m12],[m21,1-m12]]) # for asymmetric migration
    #params['beta'] = np.array([[1,M],[0,1-M]]) # for one -irectional migration
    
    """
        Test simulating trees under structured coal model
    """
    # sample_times = [10,5,0] # going from past to present with time measured in dist from present
    # sample_actions = [[2,2],[2,2],[2,2]]
    # sample_config = (sample_times,sample_actions)
    # sim_struct_tree(params,sample_config,plot=True)
    
    """
        Test lineages through time (LTT) q_k probs with or without sequential sampling
    """
    #test_ltt_probs(params)
    
    """
        Test distribution of coal times/locations for a newly sampled lineage
    """
    #test_coal_times(params)
    
    """
        Run iterative policy evaluation to compute total expected value of a policy
    """    
    sample_times = [10, 5, 0]
    sample_actions = [[2,2],[2,2],[2,2]]
    policy = (sample_times,sample_actions)
    params['k_max'] = np.sum(sample_actions)
    
    reward = iter_policy_eval(policy,params)
    #reward = reward / np.sum(reward)
    
    direct_reward = eval_policy(policy,params)
    #direct_reward = direct_reward / np.sum(direct)reward)
    
    sim_reward = monte_carlo_value(params,policy,sims=100,get_mle=False)
    #sim_reward = sim_reward / np.sum(sim_reward)
    
    print('Expected reward = ', np.sum(reward,0))
    print('Direct reward = ', direct_reward)
    print('Mean sim reward = ', sim_reward)
    
    """
         Reduced action space for testing value iteration
    """
    sample_times = [20,10,0] # should go from past to present with time measured in distance from present
    action_space = np.arange(0,11,1) # for one pop
    action_space = [[x,y] for x in action_space for y in action_space] # for both pops
    params['k_max'] = 20 # largest possible sample number + 1
    
    """
        Optimize policy using value iter algorithm
    """
    
    #Find max reward/costs
    max_sample_actions = [action_space[-1] for alt_st in range(len(sample_times))]
    policy = (sample_times,max_sample_actions)
    V = eval_policy(policy,params)
    V = np.sqrt(V[0,1]) + np.sqrt(V[1,0]) # total migration events
    max_reward = np.sum(V)
    max_cost =  np.sum(max_sample_actions) # max_sample_count
    print("Max reward =", f'{max_reward:0.4f}', ' Max cost =', f'{max_cost:0.4f}')

    opt_policy(params,action_space,sample_times)
    

    """
         Reduced action space for brute force search
    """
    # sample_times = [20,10,0] # should go from past to present with time measured in distance from present
    # action_space = np.array([0.3,0.4,0.5])
    # action_space = [[x,y] for x in action_space for y in action_space] # for both pops
    # params['k_max'] = 20 # largest possible sample number + 1

    # output_file = 'struct_coal_mdp_bruteforce_mMed_kHigh_rewards.csv'
    # brute_force_search(params,action_space,sample_times,output_file)
    
    
        