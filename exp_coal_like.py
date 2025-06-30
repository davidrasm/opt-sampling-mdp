#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:02:22 2024

Compute and optimize likelihood of a tree under the exp growth coalescent model
Coalescent prob density (likelihood) is adapted from Kuhner et al. (Genetics, 1998) 

@author: david
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import msprime

def pop_size(t,init_Ne,growth_rate):
    
    """
        Compute size of pop at some time
        Assumes exponential growth
        Input Ne is Ne at present (t=0) so Ne(t) = Ne * exp^(-growth_rate * t)
    """
    
    pop_size = init_Ne * np.exp(-growth_rate * t)
    
    return pop_size


def compute_log_like(x,tree_df,params,negative=True):
    
    """
        Computes negative log likelihood if negative=True
    """
    
    growth_rate = x # params['growth_rate']
    init_Ne = params['Ne']
    
    log_like = 0
            
    for idx, event in tree_df.iterrows():
        
        # Get time of event and time of next event
        if (idx+1 < len(tree_df)): # if not at final event
            next_time = tree_df.loc[idx+1].time
        else:
            next_time = event.time
        t_elapsed = next_time - event.time # time elapsed between events
        
        event_prob = 1.0
        prob_no_coal = 1.0
            
        if event.type == 'coalescent':
    
            Ne = pop_size(event.time, init_Ne, growth_rate)
            event_prob = 1 / Ne # the coalescent rate
            
        if t_elapsed > 0:
            k = event.ltt[0] # number of active lineages
            pairs = (k * (k-1)) / 2 # number of pairs in each pop
            
            Ne_start = pop_size(event.time, init_Ne, growth_rate)
            Ne_end = pop_size(next_time, init_Ne, growth_rate)
            Ne = (Ne_start + Ne_end) / 2 # average Ne over time intervasl
            
            prob_no_coal = np.exp(-(pairs/Ne)*t_elapsed)

        if event_prob <= 0.0:
            print('Event prob is zero or negative')
        if prob_no_coal <= 0.0:
            print('Prob no coal is zero or negative')
        log_like += np.log(event_prob) + np.log(prob_no_coal)
            
    if negative:
        log_like = -log_like
    
    #print(log_like)
    
    return log_like

def compute_log_like_kuhner(x,tree_df,params,negative=True):
    
    """
        Compute log likelihood given (corrected) density in Kuhner et al. for an exp growing pop
        Computes negative log likelihood if negative=True
    """
    
    growth_rate = x # params['growth_rate']
    init_Ne = params['Ne']
    
    log_like = 0
            
    for idx, event in tree_df.iterrows():
        
        # Get time of event and time of next event
        if (idx+1 < len(tree_df)): # if not at final event
            next_time = tree_df.loc[idx+1].time
        else:
            next_time = event.time
        t_elapsed = next_time - event.time # time elapsed between events
        
        event_prob = 1.0
        prob_no_coal = 1.0
            
        if event.type == 'coalescent':
    
            event_prob = np.exp(growth_rate * event.time) / init_Ne
            
        if t_elapsed > 0:
    
            k = event.ltt[0] # number of active lineages
            t_s = event.time # time inteval starts at previous event
            t_e = next_time # time interval ends at next event
            
            """
                Note: I added the factor of 2 in the denominator here b/c
                Kuhner et al. consider a diploid population 
            """
            total_lambda = ((k * (k-1))/(2*init_Ne * growth_rate)) * (np.exp(growth_rate * t_s) - np.exp(growth_rate * t_e))
            prob_no_coal = np.exp(total_lambda)

        if event_prob <= 0.0:
            print('Event prob is zero or negative')
        if prob_no_coal <= 0.0:
            print('Prob no coal is zero or negative')
        log_like += np.log(event_prob) + np.log(prob_no_coal)
            
    if negative:
        log_like = -log_like
    
    #print(log_like)
    
    return log_like

def build_tree_df_from_ts(ts):
         
    """
        Build dataframe for events in tree from tskit TreeSequence (ts) object
    """
    tree = ts.first()
    
    "Iterate through each event/node in tree sequence working backwards through time"
    time_list = []
    index_list = []
    child_list = []
    type_list = []
    state_list = [] # aka parent_state_list
    child_state_list = []
    children_states_list = []
    
    for u in tree.postorder():
        
        event_type = None
        if tree.is_leaf(u):
            event_type = 'sample'
        elif tree.is_internal(u):
            event_type = 'coalescent'
        else:
            print("WARNING: unknown type of event")

        children = tree.children(u)
        n_children = len(children)
        parent_state = tree.population(u)
        
        if event_type == 'coalescent':
 
            if n_children > 1:
            
                for child_idx in range(2,n_children):
                    child = children[child_idx]
                    time_list.append(tree.time(u))
                    index_list.append(-1)
                    child_list.append([child]) # -1 to represent extra node
                    type_list.append('coalescent')
                    state_list.append(parent_state)
                    child_state = tree.population(child)
                    child_state_list.append(child_state)
                    children_states_list.append([child_state])
                
                # Assuming there are two children
                child_0 = children[0]
                child_1 = children[1]
                children = [children[0], children[1]]
                child0_state = tree.population(child_0)
                child1_state = tree.population(child_1)
                child_states = [child0_state, child1_state]
    
                if child0_state != parent_state:
                    child_state = child0_state
                elif child1_state != parent_state:    
                    child_state = child1_state
                else:
                    child_state = parent_state
                    
            else:
                
                # If only one child
                child_0 = children[0]
                child_state = tree.population(child_0)
                child_states = [child_state]
                
        else:
            
            child_states = []
            child_state = 0
        
        time_list.append(tree.time(u))
        index_list.append(u)
        child_list.append(children)
        type_list.append(event_type)
        state_list.append(parent_state) # parent state
        children_states_list.append(child_states)
        child_state_list.append(child_state)
        

    # Build dataframe from event lists 
    df_dict = {'time': time_list,
               'node_index': index_list,
               'children': child_list,
               'type': type_list,
               'state': state_list,
               'child_state': child_state_list,
               'children_states': children_states_list}
    df = pd.DataFrame(df_dict)
    df = df.sort_values('time') # sort in ascending order from most recent to oldest events
    df = df.reset_index(drop=True) # need to do this or indexes will be out of order
    
    """
        Get lineages through time at each event
    """
    A = np.zeros(1) # current totals
    A_list = []
    for idx, event in df.iterrows():
        if event['type'] == 'sample':
            index = event.state
            A[index] += 1 # add sample
        else:
            # Remove all children according to their state and lineage
            for c in range(len(event.children)):
                index = event.children_states[c]
                A[index] -= 1 # remove child lineage
                
            # Add parent lineage according to state and lineage if not a dummy node added for a multifurcation
            if event.node_index >= 0:
                index = event.state
                A[index] += 1
            
        A_list.append(A.copy())
    df['ltt'] = A_list
    
    return df

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

def opt_MLE(tree_df,params,bounds):
    
    "Convert params dict to tuple"          
    #like_args = (ts, Ne, growth_rate) # args need to be passed as tuple
    like_args = (tree_df, params)
    res = minimize_scalar(compute_log_like, args=like_args, bounds=bounds, method='bounded')
    mle = res.x

    return mle

if __name__ == '__main__':
    
    """
        Test likelihood calculation with serial sampling through time
    """
    
    Ne = 50.0  # effective pop sizes
    growth_rate = 0.1 # intrinsic growth rate
    t_origin = np.abs(np.log(1 / Ne)) / growth_rate # time at which N = 1
    times = np.linspace(0, t_origin, 100) # had 40
    params = {'Ne': Ne,
               'growth_rate': growth_rate,
               'times': times}
    params['mle_bounds'] = (0.0,0.3) # only used if estimating mle values
    
    # Sequential (heterochronous) sampling
    sample_time_indexes = np.arange(0,61,20) # for 4 time points
    sample_times = list(np.flip(times[sample_time_indexes])) # going from past to present with time measured in dist from present
    sample_actions = [2,4,8,20]
    sample_config = (sample_times,sample_actions)
    ts = sim_exp_growth_tree(params,sample_config=sample_config,plot=False)
    
    # Sampling at present
    #params['samples'] = 20
    #ts = sim_exp_growth_tree(params,plot=False)
    
    tree = ts.first()
    tree_df = build_tree_df_from_ts(ts)
    #tree_df_def = build_tree_df_from_ts_defunct(ts)
    mle =  opt_MLE(tree_df,params,bounds=params['mle_bounds'])
    print("MLE: ", mle)
    mle_error = np.abs(mle - params['growth_rate'])
    print("MLE error: ", mle_error)
    
    
