# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:04:38 2021

@author: lusun
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ruspy.estimation.estimation import estimate
#from ruspy.model_code.demand_function import get_demand

import pickle


#%%
# res = pd.read_csv('D:/Google Drive/2020_WINTER/Rust/group4_new_175.csv')
# SampleSize = 37
# TimePeriod = 117

# id0 = np.kron(np.arange(SampleSize),np.ones(TimePeriod))
# id1 = np.kron(np.ones(SampleSize),np.arange(TimePeriod))


# state_new = np.array(res['state'])
# action_new = np.array(res['action'])

# delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
# delta_new[np.where(action_new==1)[0]+1] = 1
# delta_new[np.arange(SampleSize,dtype = int) * TimePeriod] = np.nan

# #%%

# # with open('data/sample_final.txt',"rb") as fp:
# #     res = pickle.load(fp)

# # SampleSize = 3000
# # TimePeriod = 100

# # id0 = np.kron(np.arange(SampleSize),np.ones(TimePeriod))
# # id1 = np.kron(np.ones(SampleSize),np.arange(TimePeriod))


# # state_new = np.array(res['res'][0]['state'])
# # action_new = np.array(res['res'][0]['action'])

# # delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
# # delta_new[np.where(action_new==1)[0]+1] = 1
# # delta_new[np.arange(SampleSize,dtype = int) * TimePeriod] = np.nan


# #%%
# # data = pd.read_pickle("group_4.pkl")
# # state_new = np.array(data['state'])
# # action_new = np.array(data['decision'])
# # delta_new = np.array(data['usage'])

# # id0 = np.array(data.index.get_level_values(0))
# # id1 = np.array(data.index.get_level_values(1))

# #%%

# # data = pd.read_pickle("rep_group_4_2500.pkl")
# # SampleSize = 37
# # TimePeriod = 117

# # id0 = np.kron(np.arange(SampleSize),np.ones(TimePeriod))
# # id1 = np.kron(np.ones(SampleSize),np.arange(TimePeriod))


# # state_new = np.array(data['state'])
# # action_new = np.array(data['decision'])

# # delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
# # delta_new[np.where(action_new==1)[0]+1] = 1
# # delta_new[np.arange(SampleSize,dtype = int) * TimePeriod] = np.nan

# #%%
# #delta_new[np.where(delta_new>3)]=4
# data_new = {'state':state_new,'decision':action_new,'usage':delta_new,'id0':id0,'id1':id1}
# dat_new = pd.DataFrame(data=data_new)
# dat_new = dat_new.set_index(['id0','id1'])
# #data.head()

# init_dict_nfxp = {
#     "model_specifications": {
#         "discount_factor": 0.9999,
#         "number_states": 175,
#         "maint_cost_func": "linear",
#         "cost_scale": 1e-3,
#     },
#     "optimizer": {
#         "approach": "NFXP",
#         "algorithm": "scipy_L-BFGS-B",
#         "gradient": "Yes",
#     },
# }

# result_transitions_nfxp, result_fixp_nfxp = estimate(init_dict_nfxp, dat_new)
# print(result_transitions_nfxp['x'], result_fixp_nfxp['x'],result_transitions_nfxp['fun'], result_fixp_nfxp['fun'],
#       result_transitions_nfxp['fun']+result_fixp_nfxp['fun'])


#%%



res = pd.read_csv('D:/Google Drive/2020_WINTER/Rust/group4_new_175.csv')
SampleSize = 37
TimePeriod = 117

# id0 = np.kron(np.arange(SampleSize),np.ones(TimePeriod))
# id1 = np.kron(np.ones(SampleSize),np.arange(TimePeriod))


# state_new = np.array(res['state'])
# action_new = np.array(res['action'])

# delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
# delta_new[np.where(action_new==1)[0]+1] = 1
# delta_new[np.arange(SampleSize,dtype = int) * TimePeriod] = np.nan


def boostrap(seed,data=res,time=TimePeriod,nbus=SampleSize):
    """
    Input:
        seed:   random seed, record of randomized ranking,
                in order to consistent with rust 1987 model
        nbus:   number of sample size, 37, in 'group4_new_175.cvs'
        time:   length of each sample path, 117, in 'group4_new_175.cvs'
        data:   data from 'gourp4_new_175.cvs'

    
    Output:
        dat_new:  keep record of new samples
    """
    #np.random.seed(seed)
    #samples = np.random.randint(0,nbus,nbus)
    samples = [i for i in range(nbus)]
    samples.remove(seed)
    samples = np.array(samples)
    
    id0 = np.kron(samples,np.ones(time))
    id1 = np.kron(np.ones(nbus-1),np.arange(time))
    
    state_new = np.zeros(time*(nbus-1))
    action_new = np.zeros(time*(nbus-1))
    print(samples)
    for i,ss in enumerate(samples):
        j = time*i
        state_new[j:j+time] = data['state'][ss*time:(ss+1)*time]
        action_new[j:j+time] = data['action'][ss*time:(ss+1)*time]
   
    delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
    delta_new[np.where(action_new==1)[0]+1] = 1
    delta_new[np.arange(nbus-1,dtype = int) * time] = np.nan
    delta_new[np.where(delta_new>3)]=3
    #data = {'state':state_new,'action':action_new}
    #dat_new = pd.DataFrame(data=data)
    ##print(samples)
    ##print(dat_new)
    return state_new,action_new,id0,id1,delta_new

#%%
# import pickle
# theta3 = np.zeros([37,4])
# theta1 = np.zeros([37,2])
# for i in range(37):
#     state_new,action_new,id0,id1,delta_new = boostrap(i,data=res,time=TimePeriod,nbus=SampleSize)
#     data_new = {'state':state_new,'decision':action_new,'usage':delta_new,'id0':id0,'id1':id1}
#     dat_new = pd.DataFrame(data=data_new)
#     dat_new = dat_new.set_index(['id0','id1'])
#     #data.head()
    
#     init_dict_nfxp = {
#         "model_specifications": {
#             "discount_factor": 0.9999,
#             "number_states": 175,
#             "maint_cost_func": "linear",
#             "cost_scale": 1e-3,
#         },
#         "optimizer": {
#             "approach": "NFXP",
#             "algorithm": "scipy_L-BFGS-B",
#             "gradient": "Yes",
#         },
#     }
    
#     result_transitions_nfxp, result_fixp_nfxp = estimate(init_dict_nfxp, dat_new)
#     theta3[i],theta1[i] = np.array(result_transitions_nfxp['x']), np.array(result_fixp_nfxp['x'])
#     print(i,result_transitions_nfxp['x'], result_fixp_nfxp['x'])

# with open('D:/Google Drive/2020_WINTER/Rust/data/seed_new/rust_rw.txt','wb') as fp:
#     pickle.dump(theta1,fp)
# with open('D:/Google Drive/2020_WINTER/Rust/data/seed_new/rust_dy.txt','wb') as fp:
#     pickle.dump(theta3,fp)
    
    
#%%
import pickle
with open('D:/Google Drive/2020_WINTER/Rust/data/seed_new/rust_rw.txt','rb') as fp:
    theta1 = pickle.load(fp)
with open('D:/Google Drive/2020_WINTER/Rust/data/seed_new/rust_dy.txt','rb') as fp:
    theta3 = pickle.load(fp)
    

def boostrap_test(seed,data=res,time=TimePeriod,nbus=SampleSize):
    """
    Input:
        seed:   random seed, record of randomized ranking,
                in order to consistent with rust 1987 model
        nbus:   number of sample size, 37, in 'group4_new_175.cvs'
        time:   length of each sample path, 117, in 'group4_new_175.cvs'
        data:   data from 'gourp4_new_175.cvs'

    
    Output:
        dat_new:  keep record of new samples
    """
    #np.random.seed(seed)
    #samples = np.random.randint(0,nbus,nbus)
    samples = seed
    
    id0 = samples*np.ones(time)
    id1 = np.arange(time)
    
    state_new = np.zeros(time)
    action_new = np.zeros(time)
    print(samples)
    state_new = (data['state'][samples*time:(samples+1)*time]).to_numpy()
    action_new = (data['action'][samples*time:(samples+1)*time]).to_numpy()
   
    delta_new = state_new - np.hstack([np.zeros(1),state_new[0:-1]])
    delta_new[np.where(action_new==1)[0]+1] = 1
    delta_new[0] = np.nan
    if len(np.where(delta_new>3)[0]) >0:
        delta_new[np.where(delta_new>3)]=3
    #data = {'state':state_new,'action':action_new}
    #dat_new = pd.DataFrame(data=data)
    ##print(samples)
    ##print(dat_new)
    return state_new,action_new,id0,id1,delta_new


def transitions_test(df,theta3): #df = data_new
    """Estimating the transition proabilities.

    The sub function for managing the estimation of the transition probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    result_transitions : dictionary
        see :ref:`result_trans`

    """
    usage = df['usage'].to_numpy(dtype=float)
    usage = usage[~np.isnan(usage)].astype(int)
    transition_count = np.array([np.sum(usage==i) for i in range(4)])#np.bincount(usage)
    
    acc1 = (np.sum(np.multiply(transition_count,theta3))+np.sum(df['decision']==1))/117
    print(acc1)
    return acc1

"""
This module contains the main function for the estimation process.
"""

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_transitions import create_transition_matrix

from ruspy.estimation.est_cost_params import accuracy_cost_params
def reward_test(init_dict, df,theta3,theta1):
    """
    Estimation function of ruspy.

    This function coordinates the estimation process of the ruspy package.

    Parameters
    ----------
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    result_cost_params : dictionary
        see :ref:`result_costs`


    """
    #transition_results = estimate_transitions(df)

    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)

    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(theta3))
    state_mat = create_state_matrix(states, num_states)
    
    alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]
    acc2 = accuracy_cost_params(np.array(theta1),
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    disc_fac,
    scale,
    alg_details,
    )
    print(acc2)
    return acc2
#%%
acc = np.zeros([37,3])
for i in range(37):
    state_new,action_new,id0,id1,delta_new = boostrap_test(i,data=res,time=TimePeriod,nbus=SampleSize)
    data_new = {'state':state_new,'decision':action_new,'usage':delta_new,'id0':id0,'id1':id1}
    dat_new = pd.DataFrame(data=data_new)
    dat_new = dat_new.set_index(['id0','id1'])
    acc[i,0] = transitions_test(dat_new,theta3[i])


    init_dict_nfxp = {
        "model_specifications": {
            "discount_factor": 0.9999,
            "number_states": 175,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "optimizer": {
            "approach": "NFXP",
            "algorithm": "scipy_L-BFGS-B",
            "gradient": "Yes",
        },
    }
    
    acc[i,1] = reward_test(init_dict_nfxp, dat_new,theta3[i],theta1[i])
    acc[i,2] = (acc[i,1] + acc[i,0])/2
    
# (array([0.43322201, 0.97635836, 0.70479018]),
#  array([0.01983952, 0.00454503, 0.01036692]))