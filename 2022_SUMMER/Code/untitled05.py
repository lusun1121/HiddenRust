# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:47:19 2020
11:44
@author: Sunlu
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize #scipy version = 1.4.1
import time as tm
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from scipy.misc import derivative
import warnings



# warnings.filterwarnings("error")
#

#%%
# def partial_derivative(func, point):
#     tol = len(point)
#     def wraps(x,var):
#         args = point.copy()#[:]
#         args[var] = x
#         #print(args)
#         return func(args)
    
#     pd = []
#     for var in range(tol):
#         wraps_new = lambda x: wraps(x,var)
#         pd.append(derivative(wraps_new,point[var]))
        
#     return 

class FullLikelihood(object):
    def __init__(self, data, Y, X, dim_x, dim_z, time, nbus, hide_state = True, disp_status = True):
        """
        A statistics workbench used to evaluate the cost parameters underlying 
        a bus replacement pattern by a forward-looking agent.
        
        Input:
            
            * data: a Pandas dataframe, which contains:
                -Y: the name of the column containing the dummy exogenous 
                    variable (here, the choice)
                -X: the name of the column containing the endogenous variable 
                    (here, the state of the bus)
            
            * dim_x: The number of belif. Start from 0 to 1.
            
            * dim_z: The number of observation state bins.
            
            * time: The length of time horizon of the observations.           
            
            * nbus: The number of buses.
            
            * hide_state: Default is True. Decide to use Hidden model or not.
            
            * disp_status: Default is False. Print the grid of opt process.
        """     
        self.alpha = 0.01
        self.data = np.array(data)
        self.endog = data.loc[:, Y].values
        self.exog = data.loc[:, X].values
        self.dim = dim_x     
        self.S = dim_z 
        self.time = time
        self.nbus = nbus
        self.hide_state = hide_state
        self.disp_status = disp_status
        
        # To speed up computations and avoid loops when computing the log 
        # likelihood, we create a few useful matrices here:
        self.N = self.endog.shape[0]
        self.Dlength = self.S*self.dim
        length = self.Dlength
        
        # A (length x 2) matrix indicating the blief of a bus at observation 
        # state i 
        self.D = [(i,j/(self.dim-1)) for i in range(self.S) 
                                     for j in range(self.dim)]
        
        # A (length x 1) matrix        
        self.oneMatrix = np.ones((length,1))
        
        # D0: A (length x N) matrix generate from D, deal with observaton states
        # D1: A (length x N) matrix generate from D, deal with belief        
        D_array = np.array(self.D).T
        self.D0 = D_array[0].reshape((length,1)) * np.ones((1,self.N))
        self.D1 = D_array[1].reshape((length,1)) * np.ones((1,self.N))
        
        # A (length x length) matrix indicating the probability of a bus 
        # transitioning from a state z to a state z' with centain belif  
        self.regen_mat = np.vstack((np.zeros((self.dim-1, length)),
                                    np.ones((1,length)),
                                    np.zeros((length-self.dim, length))))
        
        # A (2xN) matrix indicating with a dummy the decision taken by the 
        # agent for each time/bus observation (replace or maintain)        
        self.dec_mat = np.vstack(((1-self.endog), self.endog))
               
    def trans(self, theta2,p0,p1):
        """
        This function generate the transition matrix
        
        Input:
            
            * theta2: The true state transition probability
            
            * p0: The transition probability when true state s = 0 (good)
            
            * p1: The transition probability when true state s = 1 (bad)
        Output:
        
            * trans_mat: A (length x length) matrix indicating the probability 
                        of a bus transitioning from a obsevation state z to a 
                        observation state z' with centain belif in good or bad
                        true state s.
        """
        
        S = self.S
        D = self.D
        dim = self.dim
        length = self.Dlength
        
        p_0 = np.array([p0[0],p1[0]])
        p_0 = np.around(p_0,3)
        p_1 = np.array([p0[1],p1[1]])
        p_1 = np.around(p_1,3)
        p_2 = np.array([p0[2],p1[2]])
        p_2 = np.around(p_2,3)
        p_3 = 1-p_0-p_1-p_2
        
        self.P = P = np.array((p_0,p_1,p_2,p_3))
        
        trans_mat = np.zeros((length, length))
        new_x0 = lambda x: np.dot(p_0*np.array([x,1-x]),theta2) / np.dot(p_0,[x,1-x])
        new_x1 = lambda x: np.dot(p_1*np.array([x,1-x]),theta2) / np.dot(p_1,[x,1-x])
        new_x2 = lambda x: np.dot(p_2*np.array([x,1-x]),theta2) / np.dot(p_2,[x,1-x])
        new_x3 = lambda x: np.dot(p_3*np.array([x,1-x]),theta2) / np.dot(p_3,[x,1-x])
        
        matrix0 = [new_x0(i / (dim-1)) for i in range(dim)]
        matrix1 = [new_x1(i / (dim-1)) for i in range(dim)]
        matrix2 = [new_x2(i / (dim-1)) for i in range(dim)]
        matrix3 = [new_x3(i / (dim-1)) for i in range(dim)]
        
        self.matrix = x_matrix = np.vstack((matrix0,matrix1,matrix2,matrix3))
        
        for i in range(length):
            
            z_cur = D[i][0]
            x_cur = D[i][1]
            
            for j in range(4):
                
                if self.hide_state == True:
                    x_new = x_matrix[j][int(x_cur*(dim-1))]
                else:
                    x_new = 1
                    
                x_f = np.floor((self.dim-1)*x_new)/(self.dim-1)
                coe = (x_new-x_f)*(self.dim - 1)
                
                if z_cur + j < S-1:
                    ind = D.index((z_cur + j,x_f))
                    trans_mat[ind][i] = (1-coe)*np.dot(P[j],[x_cur,1-x_cur])
                    
                    if x_f != 1:
                        trans_mat[ind+1][i] = coe*np.dot(P[j],[x_cur,1-x_cur])
                        
                elif z_cur + j >= S-1:
                    ind = D.index((S-1,x_f))
                    trans_mat[ind][i] += (1-coe)*np.dot(P[j],[x_cur,1-x_cur])
                    
                    if x_f != 1:
                        trans_mat[ind+1][i] += coe*np.dot(P[j],[x_cur,1-x_cur])
                        
                else:
                    pass
                
        return trans_mat
    
    def belief(self,theta2,p0,p1, path):
        """
        This function return the belief x
        
        Input:
            
            * theta2: The true state transition probability
            
            * p0: The transition probability when s = 0
            
            * p1: The transition probability when s = 1
            
            * path: A path of mileage and replacement/maintenance record data
        
        Output:
            
            * x: A list with length 'time'. Record belief of each state z
        """
        length = self.time
        x = [1]
        p_0 = np.array([p0[0],p1[0]])
        p_1 = np.array([p0[1],p1[1]])
        p_2 = np.array([p0[2],p1[2]])
        p_3 = 1-p_0-p_1-p_2
        
        for i in range(length-1):
            
            if path[i][1] == 1:
                x_new = 1
                x.append(x_new)
                
            else:
                x_cur= np.array([x[-1],1-x[-1]])
                gap = path[i+1][0]-path[i][0]
                
                if gap == 0:
                    v1 = np.dot(p_0,x_cur)
                    v2 = np.dot(p_0,np.diag(x_cur))
                    v3 = np.dot(v2,theta2)
                    try:                    
                        x_new = v3/v1
                    except RuntimeWarning:
                        x_new = v3/(v1+np.finfo(float).eps)
                    x.append(x_new)
                    
                elif gap == 1:
                    v1 = np.dot(p_1,x_cur)
                    v2 = np.dot(p_1,np.diag(x_cur))
                    v3 = np.dot(v2,theta2)
                    try:                    
                        x_new = v3/v1
                    except RuntimeWarning:
                        x_new = v3/(v1+np.finfo(float).eps)
                    x.append(x_new)
                    
                elif gap == 2:
                    v1 = np.dot(p_2,x_cur)
                    v2 = np.dot(p_2,np.diag(x_cur))
                    v3 = np.dot(v2,theta2)
                    try:                    
                        x_new = v3/v1
                    except RuntimeWarning:
                        x_new = v3/(v1+np.finfo(float).eps)
                    x.append(x_new)
                    
                else:
                    v1 = np.dot(p_3,x_cur)
                    v2 = np.dot(p_3,np.diag(x_cur))
                    v3 = np.dot(v2,theta2)
                    try:                    
                        x_new = v3/v1
                    except RuntimeWarning:
                        x_new = v3/(v1+np.finfo(float).eps)
                    x.append(x_new)
        return x
    
    def myopic_costs(self, params):
        """
        This function computes the myopic expected cost associated with each 
        decision for each state.

        Input:

            * params: A vector, to be supplied to the maintenance linear cost 
            function. The first element of the vector is the replacement cost rc.

        Output:
            
            * A (length x 2) array containing the maintenance and replacement 
            costs for the z possible states of the bus.
        """
        
        length = self.Dlength
        D = self.D
        rc = params[0]
        thetas = np.array(params[1:])
        maint_cost = np.array([np.dot(0.001*D[d][0]*thetas,[D[d][1],1-D[d][1]])
                               for d in range(0, length)])
        repl_cost = [rc for d in range(0, length)]

        return  np.vstack((maint_cost, repl_cost)).T
     
    def fl_costs(self, params, theta2, p0, p1 ,beta=0.9999, threading = 3e-3):#beta=0.9999, threading = 0.45):#beta=0.9, threading = 1e-3):#
        """
        Compute the non-myopic expected value of the agent for each possible 
        decision and each possible state of the bus, conditional on a vector of 
        parameters and on the maintenance cost function specified at the 
        initialization of the DynamicUtility model.

        Iterates until the difference in the previously obtained expected value 
        and the new expected value is smaller than a constant 'threading'.

        Input:

            * params: A vector params for the cost function

            * theta2: Hiden state transition probability

            * p0: The transition probability when s = 0

            * p1: The transition probability when s = 1

            * beta: Default is 0.9999. A discount factor beta (optional)

            * threading: Default is 0.45. A convergence threshold (optional)

        Output:

            * EV_new: A (length x 2) array of forward-looking costs associated 
                    with each state z and each belief. Definded as Q
        """

        # Initialization of the contraction mapping
        self.EV_myopic =self.myopic_costs(params)
        EV_new = self.EV_myopic

        EV = np.zeros(EV_new.shape)        
        VError = np.max(abs(EV-EV_new))
        #print('Initial Error:',VError,np.max(abs(EV_new)))

        self.trans_mat = self.trans(theta2,p0,p1)

        # Contraction mapping Loop
        i = 0
        length = self.Dlength
        
        while(VError>threading):         

            EV = EV_new
            m = EV.min(1).reshape(length, -1)
            ecost = self.alpha*(np.log(np.exp((-EV+m)/self.alpha).sum(1) )-m.T/self.alpha+0.5772)
          
            futil_maint = np.dot(ecost, self.trans_mat)
            futil_repl = np.dot(ecost, self.regen_mat)
            futil = np.vstack((futil_maint, futil_repl)).T
            EV_new = self.EV_myopic - beta * futil

            i +=1
            VError = np.max(abs(EV-EV_new))/np.max(np.abs(EV))
            #VError = np.max(abs(EV-EV_new))
            #print(i,VError)

            if i>1000:
                break
            #print(i,VError)            
        #print('{}, Eorror:'.format(i),VError,np.max(abs(EV_new)))   

        return EV_new

    def choice_prob(self, rho, cost_array):
        """
        Returns the probability of each choice for each observed state, 
        conditional on an array of state/decision costs (generated by the 
        myopic or forward-looking cost functions)
        """
        cost_array = cost_array/rho
        cost = cost_array - cost_array.min(1).reshape(self.Dlength, -1)
        util = np.exp(-cost)
        pchoice = util / (np.sum(util, 1).reshape(self.Dlength, -1))
        # xx1,xx2 = np.where(pchoice<1e-6)[0],np.where(pchoice<1e-6)[1]
        # pchoice[xx1,xx2]=1e-6
        # pchoice[xx1,1-xx2]=1-1e-6
        return pchoice

    def loglike_full(self,theta2):
        """
        The log-likelihood of the Dynamic model(theta 2) is estimated.
        """
        self.theta2 = theta2
        self.fit_likelihood2()
        
        loglike = -self.fitted2.fun
        
        if self.disp_status:
            print(theta2,loglike)

        return -loglike
    def loglike2(self,parameters):
        theta2 = self.theta2
        
    # def loglike2(self,parameters):
    #     """
    #     The log-likelihood of the Dynamic model(theta 3) is estimated.
    #     """

    #     theta2 = parameters[6:]
        p0 = parameters[0:3]
        p1 = parameters[3:6]
        
        prob0 = np.array([p0[0],p1[0]])
        prob1 = np.array([p0[1],p1[1]])
        prob2 = np.array([p0[2],p1[2]])

        prob3 = 1 - prob0 -prob1 - prob2

        mileage = self.exog

        if self.hide_state == True:
            self.beli = []
            
            for i in range(self.nbus):
                self.beli = self.beli + self.belief(theta2,p0,p1,
                                self.data[i*self.time:(i+1)*self.time])
        else:
            self.beli = [1 for s in range(0,self.N)]
        
        logprob = 0
        for i in range(self.N-1):
            x_cur = [self.beli[i], 1-self.beli[i]]

            if mileage[i+1] - mileage[i] == 0:
                try:
                    logprob += np.log(np.dot(prob0,x_cur))
                except RuntimeWarning:
                    logprob += np.log(np.finfo(float).eps)
            
            elif mileage[i+1] - mileage[i] == 1:
                try:
                    logprob += np.log(np.dot(prob1,x_cur))
                except RuntimeWarning:
                    logprob += np.log(np.finfo(float).eps)             

            elif mileage[i+1] - mileage[i] == 2:
                try:
                    logprob += np.log(np.dot(prob2,x_cur))
                except RuntimeWarning:
                    logprob += np.log(np.finfo(float).eps)

            elif mileage[i+1] - mileage[i] == 3:
                try:
                    logprob += np.log(np.dot(prob3,x_cur))
                except RuntimeWarning:
                    logprob += np.log(np.finfo(float).eps)

        #print(f'current likelihood: {logprob}')

        return -logprob

    def loglike(self, parameters, p0, p1, theta2):
        """
        The log-likelihood of the reward model is estimated in several steps.

        1°) The current parameters are supplied to the contraction mapping 
            function

        2°) The function returns a matrix of decision probabilities for each 
            state.

        3°) This matrix is used to compute the loglikelihood of the 
            observations

        4°) The log-likelihood are then summed accross individuals, and 
            returned
        """

        params = parameters[0:3]
        rho = self.alpha

        # p0 = self.fitted2.x[0:3]
        # p1 = self.fitted2.x[3:6]
        # theta2 = self.theta2

        util = self.fl_costs(params,theta2,p0,p1) # EV
        #print(util)
        self.pchoice = pchoice =self.choice_prob(rho,util) #
        mileage = self.exog

        self.beli = []

        for i in range(self.nbus):
            self.beli = self.beli + self.belief(theta2,p0,p1,
                                        self.data[i*self.time:(i+1)*self.time])

        # A (length x N) matrix indicating the state of each observation
        b_f = np.floor((self.dim-1)*np.array(self.beli))/(self.dim-1)
        self.b_coe = (self.beli-b_f)*(self.dim-1)

        b_coe_new = self.oneMatrix * self.b_coe.reshape((1,self.N))
        mileage_new = self.oneMatrix * mileage.reshape((1,self.N))

        beli_array = np.array(self.beli).reshape((1,self.N))
        

        mat_mij = np.array(mileage_new == self.D0, dtype=int)
        beli_f = self.oneMatrix * np.floor(beli_array *(self.dim-1)) / (self.dim-1)
        beli_c = self.oneMatrix * np.ceil(beli_array * (self.dim-1)) / (self.dim-1)
        bool_to_int_f = np.array((mat_mij + np.array(beli_f == self.D1, dtype=int)) == 2, dtype=int)
        bool_to_int_c = np.array((mat_mij + np.array(beli_c == self.D1, dtype=int)) == 2, dtype=int)
        self.state_mat_f = bool_to_int_f * (1 - b_coe_new)
        self.state_mat_c = bool_to_int_c * b_coe_new
        self.state_mat = self.state_mat_f + self.state_mat_c
        

        logprob = np.log(np.dot(pchoice.T, self.state_mat))
        #print(np.sum(np.dot(pchoice.T, self.state_mat)==0))
        logprob = np.sum(self.dec_mat*logprob)
        
        if self.disp_status:
            print(parameters,logprob)

        #print(f'current likelihood:{parameters} {logprob}')

        return -logprob
    
    # def policy_gradient(self, parameters,max_iter = 100,rho = 0.01):#, p0, p1, theta2):


    #     #params = parameters[0:3]
    #     #rho = 1
    #     print("-> Matrix Generation")
    #     p0 = self.fitted2.x[0:3]
    #     p1 = self.fitted2.x[3:6]
    #     theta2 = self.fitted2.x[6:8]
    #     # p0 = [0.03920017, 0.33350266, 0.59020078]
    #     # p1 = [0.18078681, 0.75749297, 0.06072025]
    #     # theta2 = [0.94946915, 0.01199934]

    #     mileage = self.exog

    #     self.beli = []

    #     for i in range(self.nbus):
    #         self.beli = self.beli + self.belief(theta2,p0,p1,
    #                                     self.data[i*self.time:(i+1)*self.time])

    #     # A (length x N) matrix indicating the state of each observation
    #     b_f = np.floor((self.dim-1)*np.array(self.beli))/(self.dim-1)
    #     self.b_coe = (self.beli-b_f)*(self.dim-1)

    #     b_coe_new = self.oneMatrix * self.b_coe.reshape((1,self.N))
    #     mileage_new = self.oneMatrix * mileage.reshape((1,self.N))

    #     beli_array = np.array(self.beli).reshape((1,self.N))

    #     mat_mij = np.array(mileage_new == self.D0, dtype=int)
    #     beli_f = self.oneMatrix * np.floor(beli_array *(self.dim-1)) / (self.dim-1)
    #     beli_c = self.oneMatrix * np.ceil(beli_array * (self.dim-1)) / (self.dim-1)
    #     bool_to_int_f = np.array((mat_mij + np.array(beli_f == self.D1, dtype=int)) == 2, dtype=int)
    #     bool_to_int_c = np.array((mat_mij + np.array(beli_c == self.D1, dtype=int)) == 2, dtype=int)
    #     self.state_mat_f = bool_to_int_f * (1 - b_coe_new)
    #     self.state_mat_c = bool_to_int_c * b_coe_new
    #     self.state_mat = self.state_mat_f + self.state_mat_c
        

    #     Q_func = lambda params: -self.fl_costs(params,theta2,p0,p1) # EV
        
    #     ind = 1
    #     while(ind <= max_iter):
    #         print('theta1',parameters)
    #         print('-> Derivative Generation', ind)#9:54
    #         Q_val = np.exp(Q_func(parameters))
    #         Q_div = partial_derivative(Q_func, parameters)
    #         print('-> New theta update')#10:00
    #         for i in range(len(parameters)):
    #             v_div = np.sum(Q_val *Q_div[i],axis=1)/np.sum(Q_val,axis=1)
    #             pi_div = Q_div[i] - v_div.reshape([-1,1]).dot(np.ones([1,2]))
    #             l_div = self.dec_mat * np.dot(pi_div.T, self.state_mat)
    #             #l_div = np.sum(self.dec_mat*l_div)
    #             print(np.sum(l_div))
    #             parameters[i] = parameters[i] + rho*np.sum(l_div)
    #             # if i%10==0:
    #             #     i = i/2
    #         ind = ind +1
                
        
        # #self.pchoice = pchoice =self.choice_prob(rho,util) #



        
        # # if self.disp_status:
        # #     print(parameters,logprob)

        # # print(f'current likelihood:{parameters} {logprob}')

        # return parameters
    def fit(self):
        """
        estimate the belif parameter(theta 2)
        """
        # inorder to accelerate the computing speed
        # bounds = [(0.9,0.99999),(0,0.09999)]
        # self.fitted_full = opt.brute(self.loglike_full, bounds, Ns = 3) # To speed up, Ns=3
        
        self.theta2 = self.fitted_full = np.array([0.94916457, 0.01221181])

    def fit_likelihood2(self):
        """
        estimate the dynamic(theta 3)
        """
        # # bounds = [(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        # #           (0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        # #           (0,0.99999),(0,0.99999)]
        # x0 = [0.1,0.1,0.1,0.1,0.1,0.1, 0.1, 0.1]
        # cons= ({'type': 'ineq', 'fun': lambda x: x[0]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[1]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[2]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[3]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[4]-0.001},\
        #        {'type': 'ineq', 'fun': lambda x: x[5]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[6]},\
        #        {'type': 'ineq', 'fun': lambda x: x[7]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[0]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[1]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[2]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[3]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[4]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[5]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.99999-x[6]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.99999-x[7]},\
        #        {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]-x[2]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: 1-x[3]-x[4]-x[5]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: -3*x[0]-2*x[1]-x[2]+3*x[3]+2*x[4]+x[5] })
        x0 = [0.039,0.335,0.588,0.182,0.757,0.061]#[0.1,0.1,0.1,0.1,0.1,0.1]
        cons= ({'type': 'ineq', 'fun': lambda x: x[0]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: x[1]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: x[2]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: x[3]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: x[4]-0.001},\
               {'type': 'ineq', 'fun': lambda x: x[5]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[0]},\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[1]},\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[2]},\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[3]},\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[4]},\
               {'type': 'ineq', 'fun': lambda x: 0.998-x[5]},\
               {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]-x[2]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: 1-x[3]-x[4]-x[5]-0.001 },\
               {'type': 'ineq', 'fun': lambda x: -3*x[0]-2*x[1]-x[2]+3*x[3]+2*x[4]+x[5] })
        self.fitted2 = minimize(
            self.loglike2,
            x0=x0,#bounds = bounds,
            constraints= cons)#,
            #options={'disp': True})
        
    def fit_likelihood(self, x0=None, bounds=None):
        """
        estimate the reward parameter
        """
        if bounds == None:
            bounds = [(1e-6, 100),(1e-6,100),(1e-6,100)]

        if x0 == None:
            #x0 = [10,1,1,]
            #x0 =  [9.36, 0.94, 0.94] # To speed up, choose a closer initial point
            #x0 = [0.1, 0.1, 0.1]
            x0 = np.array([9.738,0.3,1.3])*self.alpha
        
        p0 = self.fitted2.x[0:3]
        p1 = self.fitted2.x[3:6]
        theta2 = self.theta2
        #theta2 = self.fitted2.x[6:8]
        #p0 = [0.03920017, 0.33350266, 0.59020078]
        #p1 = [0.18078681, 0.75749297, 0.06072025]
        #theta2 = [0.94946915, 0.01199934]
        cons= ({'type': 'ineq', 'fun': lambda x: x[2]-x[1] },\
               {'type': 'ineq', 'fun': lambda x: x[0]-1e-6 },\
               {'type': 'ineq', 'fun': lambda x: x[1]-1e-6 },\
               {'type': 'ineq', 'fun': lambda x: x[2]-1e-6 },\
               {'type': 'ineq', 'fun': lambda x: 100-x[0]},\
               {'type': 'ineq', 'fun': lambda x: 100-x[1]},\
               {'type': 'ineq', 'fun': lambda x: 100-x[2]})
        self.fitted = minimize(
            self.loglike, x0=x0,#bounds = bounds, 
            constraints= cons, args=(p0, p1, theta2))



#%%

# if __name__ == "__main__":
    
#     print('Reminder1: file group4_new_175.csv should be in the same file directory')
#     print('Reminder2: the code will take almost 4 hours')
    
#     # a dataframe has two columns, named "state" and "action".    
#     dat3 = pd.read_csv('group4_new_175.csv') 
    
#     nbus = 37
#     time = 117
    
#     #Record in form: 
#     #       theta2, logSigma, theta3, success1, ccp, [Rc,r1,r2], success2
#     data_record = [] 
#     timeStart = tm.time()
    
#     #Initialize
#     estimation = FullLikelihood(dat3,'action','state', 
#                             dim_x =51,dim_z = 175,time = time, nbus = nbus, 
#                             hide_state = True, disp_status = False)
    
#     # #theta2
#     # estimation.fit_likelihood2()
#     # data_record.append(estimation.fitted_full)          #theta2
#     # print('theta 2:\n', estimation.fitted_full)
#     # #estimation.loglike_full([0.94946915,0.01199934])
        
#     #theta3 and logSigma
#     estimation.fit_likelihood2()
#     print('theta 2:\n', estimation.fitted2.x[6:8])      #
#     data_record.append(estimation.fitted2.x[6:8])       #theta2
#     data_record.append(estimation.fitted2.fun)          #logSigma
#     data_record.append(estimation.fitted2.x[0:6])       #theta3
#     data_record.append(estimation.fitted2.success)      #success1
#     print('dynamic process:\n',estimation.fitted2.x[0:6])
    
#     #[Rc,r1,r2] and ccp
#     estimation.fit_likelihood()
#     data_record.append(estimation.fitted.fun)           #ccp
#     data_record.append(estimation.fitted.x)             #[Rc,r1,r2]
#     data_record.append(estimation.fitted.success)       #success2
#     print('reward process:\n',estimation.fitted)#2 minut
#     #
#     # print('\nTime finally use:{:.4f} hr'.format((tm.time()-timeStart)/3600))
#     # print(' theta2:{},\n logSigma:{},\n theta3:{},\n ccp:{},\
#     #       \n reward:[{}, {}, {}]'.format(data_record[0],data_record[1],
#     #       data_record[2], data_record[4], data_record[5][0],
#     #       0.001*data_record[5][1], 0.001*data_record[5][2]))
    
#     #estimation.policy_gradient([9.38559752672684, 0.7182964136458119, 1.11155388413172])


############################My Code###########################################
#import pickle
#import os
#import ray

#path = '/scratch/user/lusun8825/rust_hidden/'
#path = 'D:/Google Drive/2020_WINTER/Rust/'
dat3 = pd.read_csv('group4_new_175.csv') 

nbus = 37
time = 117

# def boostrap(seed,data=dat3,time=time,nbus=nbus):
#     """
#     Input:
#         seed:   random seed, record of randomized ranking,
#                 in order to consistent with rust 1987 model
#         nbus:   number of sample size, 37, in 'group4_new_175.cvs'
#         time:   length of each sample path, 117, in 'group4_new_175.cvs'
#         data:   data from 'gourp4_new_175.cvs'

    
#     Output:
#         dat_new:  keep record of new samples
#     """
#     np.random.seed(seed)
#     samples = np.random.randint(0,nbus,nbus)
#     state_new = np.zeros(time*nbus)
#     action_new = np.zeros(time*nbus)
#     for i,ss in enumerate(samples):
#         j = time*i
#         state_new[j:j+time] = data['state'][ss*time:(ss+1)*time]
#         action_new[j:j+time] = data['action'][ss*time:(ss+1)*time]
#     data = {'state':state_new,'action':action_new}
#     dat_new = pd.DataFrame(data=data)
#     #print(samples)
#     #print(dat_new)
#     return dat_new

# @ray.remote


# def worker_task(sed,data=dat3,time = time,nbus=nbus,path = path):
# warnings.filterwarnings("ignore")
# #seed = current_work_index
# #start = time.time()
# #for i in range(10):
# seed = sed + 1025#sed*10 + i
# print(seed)
# #if seed <0:
# #  data_new = data
# #else:
# #data_tol = []
# #for seed in seed_tol:
# #    #print(seed)
# data_new = boostrap(seed,data=data,time=time,nbus=nbus)


# #Record in form: seed theta2, logSigma, theta3, ccp, [Rc,r1,r2]
    
# #print('\n\n{} seed,\n'.format(seed))
data_record = [] 
# data_record.append(seed)

timeStart = tm.time()

#Initialize
data_new = dat3
estimation = FullLikelihood(data_new,'action','state', 
                        dim_x =51,dim_z = 175,time = time, nbus = nbus, 
                        hide_state = True, disp_status = True)

#theta2
estimation.fit()
data_record.append(estimation.fitted_full)          #theta2
print('theta 2:\n', estimation.fitted_full)
estimation.loglike_full(estimation.theta2)
data_record.append(estimation.fitted2.fun)          #logSigma
data_record.append(estimation.fitted2.x)            #theta3
print('dynamic process:\n',estimation.fitted2.x)

data_record.append(estimation.fitted2.success)      #success1
#estimation.loglike_full([0.94946915,0.01199934])
    
# #theta3 and logSigma
# estimation.fit_likelihood2()
# print('theta 2:\n', estimation.fitted2.x[6:8])      #
# data_record.append(estimation.fitted2.x[6:8])       #theta2
# data_record.append(estimation.fitted2.fun)          #logSigma
# data_record.append(estimation.fitted2.x[0:6])       #theta3
# data_record.append(estimation.fitted2.success)      #success1
# print('dynamic process:\n',estimation.fitted2.x[0:6])

#[Rc,r1,r2] and ccp
estimation.fit_likelihood()
data_record.append(estimation.fitted.fun)           #ccp
data_record.append(estimation.fitted.x)             #[Rc,r1,r2]
data_record.append(estimation.fitted.success)       #success2
print('reward process:\n',estimation.fitted)#2 minut        
print('time:',tm.time()-timeStart)
#     #Save record
#     if not os.path.exists(path+'data/seed_new'):
#         os.makedirs(path+'data/seed_new')
#     with open(path+'data/seed_new/seed{}.txt'.format(seed), "wb") as fp:   #Pickling
#         pickle.dump(data_record, fp)
#     #data_tol.append(data_record)
#     return data_record#data_tol


# #############################End#############################################
# # #if __name__ == "__main__":
# ray.shutdown()
# ray.init(num_cpus = 25)

# worker_tasks = ray.get([worker_task.remote(x) for x in range(25)])

# ray.shutdown()











# runfile('//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop/untitled05.py', wdir='//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop')
# theta 2:
#  [0.94916457 0.01221181]
# [0.94916457 0.01221181] -3656.5238063777106
# dynamic process:
#  [0.03916134 0.33333098 0.59039566 0.18078324 0.7575275  0.06068926]
# [0.09738 0.003   0.013  ] -162.6792267229877
# [0.09738001 0.003      0.013     ] -162.6792263807004
# [0.09738    0.00300001 0.013     ] -162.6792269878053
# [0.09738    0.003      0.01300001] -162.67923072530957
# \\coe-fs.engr.tamu.edu\Grads\lusun8825\Desktop\untitled05.py:501: RuntimeWarning: divide by zero encountered in log
#   logprob = np.log(np.dot(pchoice.T, self.state_mat))
# \\coe-fs.engr.tamu.edu\Grads\lusun8825\Desktop\untitled05.py:503: RuntimeWarning: invalid value encountered in multiply
#   logprob = np.sum(self.dec_mat*logprob)
# [2.30678905e+01 9.99775388e-07 9.96604877e-07] nan
# [2.39443105 0.0027001  0.0117001 ] -7894.689406088367
# [0.3270851  0.00297001 0.01287001] -676.4253492601954
# [0.12035051 0.002997   0.012987  ] -170.38920536576848
# [0.09967705 0.0029997  0.0129987 ] -162.70958256461205
# [0.09811086 0.0029999  0.01299959] -162.67181586235105
# [0.09811088 0.0029999  0.01299959] -162.6718158615755
# [0.09811086 0.00299992 0.01299959] -162.67181582929427
# [0.09811086 0.0029999  0.0129996 ] -162.67181865861778
# [4.95920584e-02 9.99996333e-07 1.00028151e-06] -193.91836772428329
# [0.09325898 0.00270001 0.01169973] -162.54378697846397
# [0.093259   0.00270001 0.01169973] -162.54378701334903
# [0.09325898 0.00270003 0.01169973] -162.543787203602
# [0.09325898 0.00270001 0.01169974] -162.54378681831827
# [8.99121032e-02 9.99999390e-07 1.14395025e-02] -162.62190191894513
# [0.09264032 0.00220111 0.01165163] -162.53944099806216
# [0.09264033 0.00220111 0.01165163] -162.5394410275629
# [0.09264032 0.00220112 0.01165163] -162.53944097740282
# [0.09264032 0.00220111 0.01165164] -162.53944089402125
# [0.09225901 0.00227711 0.01200041] -162.59649022156688
# [0.09260218 0.00220871 0.0116865 ] -162.53972781628624
# [0.09263012 0.00220314 0.01166095] -162.53939699882113
# [0.09263014 0.00220314 0.01166095] -162.53939700384163
# [0.09263012 0.00220315 0.01166095] -162.53939699783948
# [0.09263012 0.00220314 0.01166096] -162.539397004772
# [0.09256274 0.00218982 0.01164831] -162.53938546648016
# [0.09256276 0.00218982 0.01164831] -162.53938546568625
# [0.09256274 0.00218984 0.01164831] -162.53938546728625
# [0.09256274 0.00218982 0.01164833] -162.5393854655503
# reward process:
#       fun: 162.53938546648016
#      jac: array([-0.05327797,  0.05409622, -0.06240082])
#  message: 'Optimization terminated successfully'
#     nfev: 33
#      nit: 6
#     njev: 6
#   status: 0
#  success: True
#        x: array([0.09256274, 0.00218982, 0.01164831])
# time: 2160.1196212768555
