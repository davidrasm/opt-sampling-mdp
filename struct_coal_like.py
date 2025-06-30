"""
Created on Mon Feb 1st 10:03:34 2021

Compute and optimize likelihood of a tree under the structured coalescent model
Coalescent prob density (likelihood) is adapted from Volz (Genetics, 2012) and Muller et al. (MBE, 2017) 

@author: david
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import msprime
import subprocess
import pyslim
import tskit

global debug
debug = False
known_ancestral_states = False
dt_step = 1.0 # was 0.1

# Removed approximation as keyword arg to compute_log_like. Now passed with params dict
def compute_log_like(x,ts,params,negative=True,root_states=False):
    
    "x is the estimated paramter"
    
    if hasattr(x, "__len__"):
        # Estimating both migration rates
        m12 = x[0]
        m21 = x[1]
    else:
        # Estimating (symmetric) scalar migration rate
        m12 = x
        m21 = x
    
    "Set lineage state prob approximation"
    approximation = params.get('sc_approx', 'masco') # use 'masco' as default

    "Unpack params from dict"
    tree = ts.first()
    pops = 2 #ts.num_populations
    
    if 'Ne' in params:
        Ne_init = np.array(params['Ne'])
    else:
        print('Need to specify eff pop sizes Ne')
        
    if 'growth_rate' in params:
        growth_rate = params['growth_rate']
    else:
        growth_rate = [0]*pops
    
    """
        Beta is now always defined from m12 and m21
    """
    beta = np.array([[1-m21,m12],[m21,1-m12]]) # for symmetric migration
        
    flags = np.array(ts.tables.nodes.flags,ndmin=1)
    
    "Lineage arrays"
    active_lines = [] # active lines in ARG
    line_state_probs = [] # lineage state probabilities for active lines
    log_like = 0.0 # log likelihood of full tree
    
    tree_times = [tree.time(u) for u in tree.timeasc()]
    
    "Iterate through each event/node in tree sequence working backwards through time"
    #for nd, event in enumerate(ts.tables.nodes):
    for idx, nd in enumerate(tree.timeasc()):
        
        "Get time of event and time of next event"
        event_time = tree.time(nd) #event.time
        if (idx+1 < len(tree_times)): # if not at final event
            next_time = tree_times[idx+1]
        else:
            next_time = event_time
        t_elapsed = next_time - event_time # time elapsed between events
        
        # Update pop sizes Ne at time
        N0_t = Ne_init[0] * np.exp(-growth_rate[0] * event_time)
        N0_t = np.max([N0_t,1.0]) # for numerical stability
        N1_t = Ne_init[1] * np.exp(-growth_rate[1] * event_time)
        N1_t = np.max([N1_t,1.0]) # for numerical stability
        Ne = [N0_t,N1_t]
        #print('Event time = ', event_time, 'Ne = ', Ne)
        
        """
            Flags on ts.nodes:
                1 = samples
                0 = coalescent
                131072 = recombination event
                262144 = common ancestor in ARG (but no coalescence in local trees)
        """
        flag = flags[nd]
        if flag == 1:
            event_type = 'sample'
        if flag == 0:
            event_type = 'coalescent'
        if flag == 131072:
            event_type = 'recombination'
        if flag == 262144:
            event_type = 'hidden_coalescent'
        if flag == 524288:
            event_type = 'migration'
        
        "Initialize prob seeing events or no events"
        event_prob = 1.0
        prob_no_coal = 1.0
        prob_no_mig = 1.0
        
        "Update active lineages based on event type: coalescent/sampling/migration events"
        if 'sample' == event_type:
            
            #print(event_time, tree.population(nd))
            
            "Add sampled lineage"
            active_lines.append(nd)
            state_probs = np.zeros(pops)
            state_probs[tree.population(nd)] = 1.0 # set prob to 1.0 for sampled state
            line_state_probs.append(state_probs)
        
        if 'coalescent' == event_type:
    
            #print(event_time, tree.population(nd))        
    
            "Get children of parent node at coalescent event"
            coal_children = list(tree.children(nd))
            event_prob = 1.0
            
            # Check for non-standard coal events
            #num_children = len(coal_children)
            #if num_children != 2:
            #   print('Non-standard coalescent event:',str(len(coal_children)))
            
            if len(coal_children) == 1:
                
                # Replace child with paraent in arrays
                child = coal_children[0]
                child_nd = active_lines.index(child)
                active_lines[child_nd] = nd # replace child with parent
                #line_state_probs[child_nd] = line_state_probs[child_nd] # do not need to actually do this
                
            else:
            
                while len(coal_children) > 1:
                    
                    child1 = coal_children[-2]
                    child2 = coal_children[-1]
                    
                    child1_nd = active_lines.index(child1)
                    child2_nd = active_lines.index(child2)
                    
                    "Compute likelihood of coalescent event"
                    p1 = line_state_probs[child1_nd]
                    p2 = line_state_probs[child2_nd]
                    
                    # If can have births between pops
                    _lambdas = np.zeros((pops,pops))
                    for k in range(pops):
                        for l in range(pops):
                            if k == l:
                                _lambdas[k,l] = 2 * p1[k] * p2[l] * beta[k,l] / Ne[k]
                            else:
                                _lambdas[k,l] = (p1[k] * p2[l] + p1[l] * p2[k]) * beta[k,l] / Ne[k]
                        
                    lambda_sum = np.sum(_lambdas)
                    event_prob *= lambda_sum
                    
                    "Compute new parent state probs"
                    if known_ancestral_states:
                        parent_probs = np.zeros(pops)
                        parent_probs[tree.population(nd)] = 1.0
                    else:
                        parent_probs = np.sum(_lambdas,axis=1) / lambda_sum # renormalize probs
                    
                    # if np.isnan(parent_probs).any(): 
                    #      print('Warning: Parent probs are NaN')
                         
                    # if len(np.nonzero(parent_probs)[0]) < 2:
                    #     print('Warning: Zeros in parent lineage state probs')
                    
                    "Update lineage arrays - overwriting child1 with parent"
                    #print('State probs before coal: ', line_state_probs, 'Time: ', event_time)
                    if len(coal_children) == 2:
                        active_lines[child1_nd] = nd # name of parent
                    line_state_probs[child1_nd] = parent_probs
                    del active_lines[child2_nd]
                    del line_state_probs[child2_nd]
                    del coal_children[-1]
                    #print('State probs after coal: ', line_state_probs, 'Time: ', event_time)
        
        if 'single_coalescent' == event_type:
            
            """
                Defunct: replaced by if 'coalesent' code above but keeping for comparison
            """
            
            "Get children of parent node at coalescent event"
            coal_children = tree.children(nd)
            
            "Make sure coalescent events only occur among two lineages"
            if len(coal_children) > 2:
                print("ERROR: Parent has more than two children at coalescent node")
            assert len(coal_children) == 2
            child1 = coal_children[0]
            child2 = coal_children[1]
            
            child1_nd = active_lines.index(child1)
            child2_nd = active_lines.index(child2)
                
            "Compute likelihood of coalescent event"
            p1 = line_state_probs[child1_nd]
            p2 = line_state_probs[child2_nd]
            
            # If can have births between pops
            _lambdas = np.zeros((pops,pops))
            for k in range(pops):
                for l in range(pops):
                    if k == l:
                        _lambdas[k,l] = 2 * p1[k] * p2[l] * beta[k,l] / Ne[k]
                    else:
                        _lambdas[k,l] = (p1[k] * p2[l] + p1[l] * p2[k]) * beta[k,l] / Ne[k]
                
            lambda_sum = np.sum(_lambdas)
            event_prob = lambda_sum
            
            "Compute new parent state probs"
            if known_ancestral_states:
                parent_probs = np.zeros(pops)
                parent_probs[tree.population(nd)] = 1.0
            else:
                parent_probs = np.sum(_lambdas,axis=1) / lambda_sum # renormalize probs
            
            "Update lineage arrays - overwriting child1 with parent"
            active_lines[child1_nd] = nd # name of parent
            line_state_probs[child1_nd] = parent_probs
            del active_lines[child2_nd]
            del line_state_probs[child2_nd]
                
        if 'migration' == event_type:
            
            child = tree.children(nd)[0]
            child_idx = active_lines.index(child)
            active_lines[child_idx] = nd # name of parent
            
            "Handling of migration events as tree nodes not handled in this model"
            event_prob = 1.0 # pretend as if we don't see migration events
            
        "Integrate lineage prob equations backwards" 
        
        "Compute prob of no coalescent over time interval"
        if not np.isclose(t_elapsed, 0):
            
            if known_ancestral_states:
                
                A = np.zeros(pops)
                for probs in line_state_probs: A += probs # sum line probs to get total number of lines in each state
                
                "Compute prob of no coalescent over time interval"
                pairs = (A * (A-1)) / 2 # number of pairs in each pop
                lambdas =  pairs * (1/Ne) # coal rate in each pop   
                prob_no_coal = np.exp(-np.sum(lambdas)*t_elapsed)
            
                "Compute prob of no migration over the time interval"
                sam = 0
                for i in range(pops):
                    for z in range(pops):
                        sam += (A[i])*(M[i][z])
                prob_no_mig = np.exp(-sam*t_elapsed)
                
            else:
                
                "Integrate lineage prob equations backwards"
                dt_times = list(np.arange(event_time,next_time,dt_step)) # integration steps going backwards in time
                for t_idx, tx in enumerate(dt_times):
                    
                    "Fix this so index does not go out of bounds"
                    if (t_idx+1 < len(dt_times)):
                        dt = dt_times[t_idx+1] - tx # integration time step
                    else:
                        dt = next_time - tx
                    if debug:
                        print("dttimes",tx,dt,event_time,next_time)
                    
                    #print('State probs before update: ', line_state_probs, 'Time: ', dt_times[t_idx])
                    line_state_probs = update_line_state_probs(line_state_probs,Ne,beta,dt,approximation=approximation)
                    #print('State probs after update: ', line_state_probs, 'Time: ', dt_times[t_idx])
                    
                    A = np.zeros(pops)
                    for probs in line_state_probs: A += probs # sum line probs to get total number of lines in each state
                    
                    lambda_sum = 0
                    for k in range(pops):
                        for l in range(pops):
                            if k == l:
                                lambda_sum += A[k] * (A[k]-1) * beta[k,k] / (2*Ne[k])
                            else:
                                lambda_sum += A[k] * A[l] * beta[k,l] / Ne[k]
                    
                    "Compute prob of no migration over the time interal"
                    prob_no_coal *= np.exp(-np.sum(lambda_sum)*dt)
                    prob_no_mig = 1.0
        
        log_like += np.log(event_prob) + np.log(prob_no_coal) + np.log(prob_no_mig)
        
        #print('State probs: ', line_state_probs, 'Time: ', next_time)
    
    if negative:
        log_like = -log_like    
    
    if root_states:
        print('State probs: ', line_state_probs)
        return np.mean(line_state_probs,axis=0)
    else:
        return log_like


def update_line_state_probs(line_state_probs,Ne,beta,dt,approximation='masco'):

    A = np.zeros(2)
    for probs in line_state_probs: A += probs # sum line probs to get total number of lines in each state    

    if approximation == 'naive':
        
        """
            Compute migration rates "naively" without any corrections
            This appears to bias estimated migration rates upwards so not using
        """
        
        mig_01 = beta[0,1] / Ne[0]
        mig_10 = beta[1,0] / Ne[1]
        M_matrix = np.array([[0,mig_10],[mig_01,0]]) # make M into a rate matrix -- matrix is transposed to reflect backwards rates 
        Q = M_matrix - np.diag(np.sum(M_matrix,axis=1)) # set diagonals to negative row sums
        expQdt = expm(Q*dt) # exponentiate time-scaled transition rate matrix

        "Update line state probs using Euler integration"
        for ldx,probs in enumerate(line_state_probs):
            line_state_probs[ldx] = np.matmul(probs,expQdt)
        
        
    elif approximation == 'volz':  

        """
            Compute migration rates using Volz finite size correction (Ne - A) 
            This appears to bias estimated migration rates downwards so not using
        """
        
        mig_01 = (Ne[0] - A[0]) * beta[0,1] / Ne[0] # mig from 0 -> 1 in forwards time
        mig_10 = (Ne[1] - A[1]) * beta[1,0] / Ne[1] # mig from 1 -> 0 in forwards time
        M_matrix = np.array([[0,mig_10],[mig_01,0]]) # make M into a rate matrix -- matrix is transposed to reflect backwards rates 
        Q = M_matrix - np.diag(np.sum(M_matrix,axis=1)) # set diagonals to negative row sums
        expQdt = expm(Q*dt) # exponentiate time-scaled transition rate matrix

        "Update line state probs using Euler integration"
        for ldx,probs in enumerate(line_state_probs):
            line_state_probs[ldx] = np.matmul(probs,expQdt)
            
    elif approximation == 'hybrid':  

        """
            Hybrid approach that's a compromise between naive and volz approx
        """
        mig_01_naive = beta[0,1] / Ne[0]
        mig_10_naive = beta[1,0] / Ne[1]
        mig_01 = (Ne[0] - A[0]) * beta[0,1] / Ne[0] # mig from 0 -> 1 in forwards time
        mig_10 = (Ne[1] - A[1]) * beta[1,0] / Ne[1] # mig from 1 -> 0 in forwards time
        mig_01 = mig_01_naive + mig_01 / 2
        mig_10 = mig_10_naive + mig_10 / 2
        
        M_matrix = np.array([[0,mig_10],[mig_01,0]]) # make M into a rate matrix -- matrix is transposed to reflect backwards rates 
        Q = M_matrix - np.diag(np.sum(M_matrix,axis=1)) # set diagonals to negative row sums
        expQdt = expm(Q*dt) # exponentiate time-scaled transition rate matrix

        "Update line state probs using Euler integration"
        for ldx,probs in enumerate(line_state_probs):
            line_state_probs[ldx] = np.matmul(probs,expQdt)
            
    elif approximation == 'masco':
        
        """
            This approximation works better with symmetric migration but does lead to a small upwards bias
        """
        mig_01 = beta[0,1] / Ne[0]
        mig_10 = beta[1,0] / Ne[1]
        
        "Update line state probs using Euler integration"
        for ldx,probs in enumerate(line_state_probs):
            new_probs = np.zeros(2)
            new_probs[0] += probs[0] + (probs[1] * mig_01 - probs[0] * mig_10 - probs[0] * ((A[0] * beta[0,0] / Ne[0]) + (A[1] * beta[1,0] / Ne[1]))) * dt
            new_probs[1] += probs[1] + (probs[0] * mig_10 - probs[1] * mig_01 - probs[1] * ((A[1] * beta[1,1] / Ne[1]) + (A[0] * beta[0,1] / Ne[0]))) * dt
            new_probs[new_probs < 0] = 0
            
            # if np.isnan(new_probs).any(): 
            #     print('Warning: Updated probs are NaN')
            
            # if len(np.nonzero(new_probs)[0]) < 2:
            #     print('Warning: Zeros in updated lineage state probs')
            
            line_state_probs[ldx] = new_probs / np.sum(new_probs) # renormalize probs
            
    elif approximation == 'masco-volz':
        
        """
            Not sure why but this correction seems to work best with asymmetric migration
        """
        
        mig_01 = (Ne[0] - A[0]) * beta[0,1] / (2*Ne[0])
        mig_10 = (Ne[1] - A[1]) * beta[1,0] / (2*Ne[1])
    
        "Update line state probs using Euler integration"
        for ldx,probs in enumerate(line_state_probs):
            new_probs = np.zeros(2)
            new_probs[0] += probs[0] + (probs[1] * mig_01 - probs[0] * mig_10 - probs[0] * ((A[0] * beta[0,0] / Ne[0]) + (A[1] * beta[1,0] / Ne[1]))) * dt
            new_probs[1] += probs[1] + (probs[0] * mig_10 - probs[1] * mig_01 - probs[1] * ((A[1] * beta[1,1] / Ne[1]) + (A[0] * beta[0,1] / Ne[0]))) * dt
            new_probs[new_probs < 0] = 0
            
            line_state_probs[ldx] = new_probs / np.sum(new_probs) # renormalize probs
        
    else:
        
        print("Approximation for updating line state probs not recognized")

    return line_state_probs

def sim_struct_tree(params,sample_config=None,plot=False):
    
    """
        Simulate under island (metapopulation) demographic model with possilbe exponential growth
        See here how to sample serially through time:
            https://tskit.dev/msprime/docs/stable/ancestry.html#sampling-time
        Important: Init pop sizes are sizes at present such that size of population is se^{rt} at time t in the past
    """
    
    init_Ne = params['Ne']
    growth_rate = params['growth_rate']
    M = params['M']
    
    demography = msprime.Demography.island_model(init_Ne, M, growth_rate=growth_rate)
    
    if sample_config:
    
        # Sample at multiple time points
        sample_times, sample_sizes = sample_config
        
        sample_sets = []
        for time, sizes in zip(sample_times, sample_sizes):
            sample_sets.append(msprime.SampleSet(sizes[0], time=time, population=0))
            sample_sets.append(msprime.SampleSet(sizes[1], time=time, population=1))
                
        ts = msprime.sim_ancestry(samples=sample_sets, ploidy=1, demography=demography, additional_nodes=(msprime.NodeType.MIGRANT),coalescing_segments_only=False)
        
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

def sample_slim_tree(ts, sample_config):
    
    "Sample Fraction of individuals in each generation and keep only haploid"
    keep_nodes = []
    data = []

    sample_times, sample_actions = sample_config # sample_times need to correspond to generations in backwards time

    for time_idx, g in enumerate(sample_times):
        
        #print("generation: ", g)
        #print("individuals: ", inds)
        
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
    ts = ts.simplify(keep_nodes, keep_input_roots=True)
    
    return ts

def like_profile_Ne(ts,params,true_val):
    
    profile_vals = list(np.arange(20,200,5))
    like_vals = []
    for val in profile_vals:
        like_vals.append(compute_log_like(val,ts,params,negative=False))
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 1)
    sns.lineplot(x=profile_vals, y = like_vals, ax=axs)
    axs.plot([true_val,true_val],axs.get_ylim(), 'darkred')
    axs.set_xlabel('Ne', fontsize=14)
    axs.set_ylabel('Likelihood', fontsize=14)
    fig.set_size_inches(6, 6)
    fig.savefig('struct_coal_MLE_Ne_test.png', dpi=200)
    
def like_profile_r(ts,params,true_val):
    
    profile_vals = list(np.arange(0.0,0.2,0.01))
    like_vals = []
    for val in profile_vals:
        like_vals.append(compute_log_like(val,ts,params,negative=False))
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 1)
    sns.lineplot(x=profile_vals, y = like_vals, ax=axs)
    axs.plot([true_val,true_val],axs.get_ylim(), 'darkred')
    axs.set_xlabel('Growth rate r', fontsize=14)
    axs.set_ylabel('Likelihood', fontsize=14)
    fig.set_size_inches(6, 6)
    fig.savefig('struct_coal_MLE_growthrate_test.png', dpi=200)

def like_profile_M(ts,params,true_val):
    
    profile_vals = list(np.arange(0.01,0.5,0.01))
    like_vals = []
    for val in profile_vals:
        like_vals.append(compute_log_like(val,ts,params,negative=False))
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 1)
    sns.lineplot(x=profile_vals, y = like_vals, ax=axs)
    axs.plot([true_val,true_val],axs.get_ylim(), 'darkred')
    axs.set_xlabel('Migration rate', fontsize=14)
    axs.set_ylabel('Likelihood', fontsize=14)
    fig.set_size_inches(6, 6)
    fig.savefig('struct_coal_MLE_migRate_test.png', dpi=200)
    
def opt_MLE(ts,params,bounds):
    
    """
        Optimize MLE for a single scalar param
    """
    
    "Convert params dict to tuple"          
    #like_args = (ts, Ne, growth_rate) # args need to be passed as tuple
    like_args = (ts, params)
    res = minimize_scalar(compute_log_like, args=like_args, bounds=bounds, method='bounded')
    mle = res.x

    return mle

def opt_MLE_params(ts,params,bounds):
    
    """
        Optimize MLE with more than one param
        x0: init parmas
    """
    
    "Convert params dict to tuple"
    x0 = [params['m12'],params['m21']]          
    like_args = (ts, params)
    res = minimize(compute_log_like, x0, args=like_args, bounds=bounds, method='Nelder-Mead')
    mle = res.x

    return mle

if __name__ == '__main__':
       
    Ne = [50,50]
    #growth_rate = [0.1,0.1] # for exp growth
    growth_rate = [0.,0.] # for constant sized pops
    M = 0.1 # migration rate
    m12 = M # migration rate from 1->2
    m21 = M # migration rate from 2->1
    #t_origin = np.abs(np.log(1 / Ne[0])) / growth_rate[0] # time at which N = 1
    t_origin = 2*Ne[0] # for constant sized pops
    times = np.linspace(0, 2*t_origin, 100) # had 40
    params = {'Ne': Ne,
               'growth_rate': growth_rate,
               'm12': m12,
               'm21': m21,
               'times': times}
    params['migration'] = 'asymmetric'
    
    """
        Simulate trees under structured WF model in msprime
    """
    #sample_times = [20,10,0] # going from past to present with time measured in dist from present
    #sample_actions = [[20,20],[20,20],[20,20]]
    #sample_times = [0] # going from past to present with time measured in dist from present
    #sample_actions = [[20,20]]
    #sample_config = (sample_times,sample_actions)
    #ts = sim_struct_tree(params,sample_config,plot=False)
    
    """
        Simulate trees under structured WF model in SLiM
    """
    sim_script = "./WF_assymetric_migration.slim"
    generations = 100
    slim_Ne = int(Ne[0] / 2) # SLiM assumes diploid pops, so setting to half of diploid size
    ne_param_string = "ne=" + str(slim_Ne)
    #m_param_string = "m=" + str(M)
    m12_param_string = "monetwo=" + str(m12)
    m21_param_string = "mtwoone=" + str(m21) # set to zero for assymetric
    g_param_string = "g=" + str(generations)
    subprocess.run(["slim", "-d", ne_param_string, "-d", m12_param_string, "-d", m21_param_string, "-d", g_param_string, sim_script])
    tree = tskit.load("./two_pop_migration.trees")
    
    #sample_times = [20,10,0] # going from past to present with time measured in dist from present
    #sample_actions = [[20,20],[20,20],[20,20]]
    sample_times = [0]
    sample_actions = [[10,10]]
    sample_config = (sample_times,sample_actions)
    ts = sample_slim_tree(tree, sample_config)
    
    "Check likelihood is valid"
    #L = compute_log_like(M,ts,params,negative=False)
    #print('Log like = ', str(L))
    
    "Test likelihood profiles for params"
    #like_profile_Ne(ts,params,Ne[0])
    #like_profile_r(ts,params,growth_rate[0])
    #like_profile_M(ts,params,M)
    
    "Test optimization of likelihood function with single scalar params"
    #params['mle_bounds'] = (0.001,0.25) # only used if estimating mle values
    #mle = opt_MLE(ts,params,bounds=params['mle_bounds'])
    #print("MLE: ", mle)
    
    params['mle_bounds'] = ((0.001,0.25),(0.001,0.25))
    mle = opt_MLE_params(ts,params,bounds=params['mle_bounds'])
    print("MLE: ", mle)
    

