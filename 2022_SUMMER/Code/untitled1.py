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
        self.alpha = 1
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
        #print(i,VError/np.max(abs(EV)))            
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
        util = np.exp(-cost/self.alpha)
        pchoice = util / (np.sum(util, 1).reshape(self.Dlength, -1))

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
        rho = 1

        # p0 = self.fitted2.x[0:3]
        # p1 = self.fitted2.x[3:6]
        # theta2 = self.theta2

        util = self.fl_costs(params,theta2,p0,p1) # EV
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



# runfile('//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop/untitled025.py', wdir='//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop')
# theta 2:
#  [0.94916457 0.01221181]
# [0.94916457 0.01221181] -3656.5238063777106
# dynamic process:
#  [0.03916134 0.33333098 0.59039566 0.18078324 0.7575275  0.06068926]
# [9.738 0.3   1.3  ] -469.81329243715584
# [9.73800001 0.3        1.3       ] -469.81329431932335
# [9.738      0.30000001 1.3       ] -469.8132899533001
# [9.738      0.3        1.30000001] -469.8132833705977
# [1.01145794e-06 1.00000000e+02 1.00000000e+02] -214819.87146217749
# [ 8.44012475 13.58796222 14.45468276] -13171.206088395418
# [9.44612094 3.28832873 4.25835552] -1439.8662279712873
# [9.63489475 1.35561664 2.34502871] -531.9107540928709
# [9.6900568  0.79085414 1.78593083] -400.9158135770032
# [9.69005681 0.79085414 1.78593083] -400.9158136377171
# [9.6900568  0.79085415 1.78593083] -400.91581285722566
# [9.6900568  0.79085414 1.78593084] -400.91581599381976
# [1.00003159e-06 2.58448044e-01 2.58448044e-01] -3545.1395258611615
# [8.72105122 0.73761353 1.63318255] -373.78533323071593
# [8.72105123 0.73761353 1.63318255] -373.78533326148295
# [8.72105122 0.73761354 1.63318255] -373.7853325330333
# [8.72105122 0.73761353 1.63318256] -373.78533577302073
# [9.99987250e-07 1.43053117e-01 1.43053117e-01] -3302.14585811132
# [7.8489462  0.67815749 1.4841696 ] -349.46718021364154
# [7.84894621 0.67815749 1.4841696 ] -349.46718022571275
# [7.8489462 0.6781575 1.4841696] -349.4671795327387
# [7.8489462  0.67815749 1.48416962] -349.4671828328412
# [9.99992758e-07 3.31676334e-02 3.31676334e-02] -3070.5625854649766
# [7.06405168 0.6136585  1.33906941] -326.3621710953351
# [7.06405169 0.6136585  1.33906941] -326.36217110254364
# [7.06405168 0.61365852 1.33906941] -326.3621704244021
# [7.06405168 0.6136585  1.33906942] -326.3621737176534
# [1.00003861e-06 1.00000510e-06 1.00000687e-06] -3000.6138751865337
# [6.35764661 0.55229275 1.20516257] -305.306322663401
# [6.35764662 0.55229275 1.20516257] -305.3063226709224
# [6.35764661 0.55229277 1.20516257] -305.30632200094675
# [6.35764661 0.55229275 1.20516258] -305.3063252599464
# [9.99482010e-07 9.99861439e-07 9.99919697e-07] -3000.6138869378365
# [5.72188205 0.49706358 1.08464641] -287.17061295627536
# [5.72188206 0.49706358 1.08464641] -287.1706129660565
# [5.72188205 0.49706359 1.08464641] -287.17061230379585
# [5.72188205 0.49706358 1.08464642] -287.1706155274168
# [1.01572344e-06 1.00303823e-06 1.00266035e-06] -3000.6135437855664
# [5.14969395 0.44735732 0.97618187] -270.34401740084166
# [5.14969396 0.44735732 0.97618187] -270.34401740788735
# [5.14969395 0.44735733 0.97618187] -270.3440167582774
# [5.14969395 0.44735732 0.97618188] -270.34401995686585
# [3.67628211e-02 1.00000193e-06 1.00000175e-06] -2289.7618617720336
# [4.63840083 0.40262169 0.87856378] -255.3748094159443
# [4.63840085 0.40262169 0.87856378] -255.3748094205273
# [4.63840083 0.4026217  0.87856378] -255.37480878466374
# [4.63840083 0.40262169 0.8785638 ] -255.37481195173334
# [7.33876400e-02 1.00000619e-06 1.00000408e-06] -1721.5004474868308
# [4.18189951 0.36235962 0.7907075 ] -242.07860995677873
# [4.18189953 0.36235962 0.7907075 ] -242.07860995942212
# [4.18189951 0.36235963 0.7907075 ] -242.07860933875173
# [4.18189951 0.36235962 0.79070752] -242.07861246501704
# [1.04165214e-01 1.00000388e-06 1.00000186e-06] -1342.7265617686453
# [3.77412608 0.32612376 0.71163685] -230.2911798947364
# [3.7741261  0.32612376 0.71163685] -230.29117989572399
# [3.77412608 0.32612377 0.71163685] -230.29117929282475
# [3.77412608 0.32612376 0.71163687] -230.291182368145
# [1.33019886e-01 9.99991715e-07 9.99996461e-07] -1059.5564781416488
# [3.41001546 0.29351148 0.64047327] -219.60483562020514
# [3.41001548 0.29351148 0.64047327] -219.60483561818498
# [3.41001546 0.2935115  0.64047327] -219.6048350392482
# [3.41001546 0.29351148 0.64047328] -219.6048380461905
# [1.93047103e-01 9.99998237e-07 9.99999313e-07] -650.1759541947296
# [3.08831863 0.26416043 0.57642604] -224.26240090968176
# [3.30307581 0.2837545  0.61918245] -237.09282945378231
# [3.3993215  0.29253578 0.63834419] -223.70573554029784
# [3.40894607 0.29341391 0.64026036] -219.57412305280735
# [3.40894608 0.29341391 0.64026036] -219.5741230507988
# [3.40894607 0.29341393 0.64026036] -219.57412247191542
# [3.40894607 0.29341391 0.64026037] -219.5741254785658
# [1.25862369e-01 1.00015894e-06 1.00007400e-06] -1123.8515751067562
# [3.0806377  0.26407262 0.57623442] -224.02838401965016
# [3.29820609 0.28351695 0.61866411] -239.10376879554198
# [3.39787207 0.29242422 0.63810073] -224.75492457931423
# [3.40783867 0.29331494 0.6400444 ] -219.54297497705863
# [3.40783868 0.29331494 0.6400444 ] -219.54297497503913
# [3.40783867 0.29331496 0.6400444 ] -219.54297439624034
# [3.40783867 0.29331494 0.64004441] -219.54297740269686
# [1.95736055e-01 1.00002482e-06 1.00001021e-06] -636.4870085449572
# [3.08662841 0.26398355 0.57604006] -224.2011378876333
# [3.30109386 0.28356751 0.61877443] -238.14241391511814
# [3.39716419 0.2923402  0.6379174 ] -225.747415983182
# [3.40677122 0.29321747 0.6398317 ] -219.51229636460917
# [3.40677123 0.29321747 0.6398317 ] -219.5122963626018
# [3.40677122 0.29321748 0.6398317 ] -219.5122957838552
# [3.40677122 0.29321747 0.63983171] -219.51229879001605
# [1.26448084e-01 1.00011790e-06 1.00005421e-06] -1118.4487012524844
# [3.07873891 0.26389582 0.57584863] -223.96104569068967
# [3.29610859 0.28332573 0.61824683] -239.0197711441392
# [3.39570496 0.29222829 0.63767321] -226.61357573888287
# [3.40566459 0.29311855 0.63961585] -219.48116874005385
# [3.40566461 0.29311855 0.63961585] -219.481168738036
# [3.40566459 0.29311856 0.63961585] -219.48116815937416
# [3.40566459 0.29311855 0.63961586] -219.4811711653399
# [1.92437129e-01 1.00009052e-06 1.00003787e-06] -653.3291622744446
# [3.08434185 0.26380679 0.57565436] -224.12211088606003
# [3.29877865 0.28336818 0.61833947] -239.08750924902233
# [3.394976   0.29214351 0.63748821] -227.36400096288926
# [3.40459573 0.29302105 0.63940309] -219.45048503220042
# [3.40459575 0.29302105 0.63940309] -219.45048503019368
# [3.40459573 0.29302106 0.63940309] -219.45048445158577
# [3.40459573 0.29302105 0.6394031 ] -219.45048745726035
# [1.30357950e-01 9.99992925e-07 9.99996751e-07] -1083.0317559210089
# [3.07717196 0.26371904 0.57546288] -223.9035552802057
# [3.29420185 0.28314161 0.61784507] -239.8747180330985
# [3.39355635 0.2920331  0.63724728] -227.94050278295558
# [3.4034918  0.29292225 0.63918751] -219.1539094546596
# [3.40349181 0.29292225 0.63918751] -219.1539094511045
# [3.4034918  0.29292227 0.63918751] -219.15390887490418
# [3.4034918  0.29292225 0.63918752] -219.15391187517574
# [3.597088   0.53472561 0.53472561] -209.01236700233653
# [3.59708802 0.53472561 0.53472561] -209.01236737589312
# [3.597088   0.53472563 0.53472561] -209.01236678970855
# [3.597088   0.53472561 0.53472563] -209.01236694006383
# [3.24705937 0.35623565 0.46620656] -209.34747021654988
# [3.43141743 0.45024528 0.50229515] -243.4694671931981
# [3.58052094 0.52627758 0.53148257] -209.091071163246
# [3.59061583 0.53142526 0.53345867] -208.90255294127707
# [3.59061584 0.53142526 0.53345867] -208.9025533139686
# [3.59061583 0.53142527 0.53345867] -208.90255272792012
# [3.59061583 0.53142526 0.53345868] -208.9025528841764
# [1.81899404e+00 1.00000000e-06 1.73305353e-01] -204.95564711602566
# [1.81899405e+00 1.00000000e-06 1.73305353e-01] -204.9556482159087
# [1.81899404e+00 1.01490116e-06 1.73305353e-01] -204.95564562787834
# [1.81899404e+00 1.00000000e-06 1.73305368e-01] -204.95564254612665
# [9.99969287e-07 2.33648088e-02 2.33648088e-02] -3049.8913973490903
# [1.63709473 0.00233738 0.1583113 ] -197.45410766101034
# [1.63709475 0.00233738 0.1583113 ] -197.45410872524695
# [1.63709473 0.0023374  0.1583113 ] -197.45410638317318
# [1.63709473 0.00233738 0.15831131] -197.45410383614805
# [7.41605773e-07 1.81152998e-02 1.81152729e-02] -3038.826322256518
# [1.47338533 0.00391517 0.1442917 ] -189.6399875605986
# [1.47338535 0.00391517 0.1442917 ] -189.63998874901412
# [1.47338533 0.00391519 0.1442917 ] -189.63998624281044
# [1.47338533 0.00391517 0.14429171] -189.63998296891089
# [0.00020337 0.0194834  0.01950311] -3037.4205796659608
# [1.32606714 0.005472   0.13181284] -181.66070670402416
# [1.32606715 0.005472   0.13181284] -181.66070798181008
# [1.32606714 0.00547201 0.13181284] -181.66070533750775
# [1.32606714 0.005472   0.13181285] -181.66070131954837
# [-0.05923336  0.02049698  0.01482481] -4486.034283058455
# [1.18753709 0.00697449 0.12011403] -174.11461441952866
# [1.1875371  0.00697449 0.12011403] -174.11461562403196
# [1.18753709 0.00697451 0.12011403] -174.11461315275076
# [1.18753709 0.00697449 0.12011405] -174.11460907675564
# [20.87446479  8.0088712  18.12542735] -9516.344203092442
# [4.85376272 1.49713889 3.47317867] -1693.0879563612184
# [1.92936647 0.30849651 0.79857797] -371.01243667175163
# [1.3784931  0.08458997 0.294759  ] -180.76003460035818
# [1.2725149  0.04151435 0.19783323] -166.31071807499174
# [1.27251492 0.04151435 0.19783323] -166.31071816012894
# [1.2725149  0.04151437 0.19783323] -166.31071779818893
# [1.2725149  0.04151435 0.19783324] -166.3107189211499
# [0.76749417 0.06451617 0.09256286] -166.6239961570351
# [1.02824155 0.05264009 0.14691503] -163.3028367999554
# [1.02824156 0.05264009 0.14691503] -163.30283666289662
# [1.02824155 0.05264011 0.14691503] -163.30283695079473
# [1.02824155 0.05264009 0.14691504] -163.30283777671676
# [8.37669907e-01 1.00000000e-06 1.06734336e-01] -162.93771929121795
# [8.37669921e-01 1.00000000e-06 1.06734336e-01] -162.93771906085976
# [8.37669907e-01 1.01490116e-06 1.06734336e-01] -162.9377193798153
# [8.37669907e-01 1.00000000e-06 1.06734351e-01] -162.93771983746828
# [0.96469558 0.04940248 0.11221868] -162.88222892770884
# [0.9036762  0.02567147 0.10958416] -162.58881096802835
# [0.90367622 0.02567147 0.10958416] -162.58881096380497
# [0.9036762  0.02567148 0.10958416] -162.58881101312042
# [0.9036762  0.02567147 0.10958418] -162.58881078877818
# [0.93780856 0.01843165 0.12116433] -162.56306606795465
# [0.93780857 0.01843165 0.12116433] -162.56306606026797
# [0.93780856 0.01843166 0.12116433] -162.56306604233336
# [0.93780856 0.01843165 0.12116434] -162.5630662186606
# [0.92603871 0.02171353 0.11668792] -162.5394343691621
# [0.92603872 0.02171353 0.11668792] -162.53943436823457
# [0.92603871 0.02171355 0.11668792] -162.53943436837645
# [0.92603871 0.02171353 0.11668794] -162.53943437776263
# [0.92561723 0.02187332 0.11648356] -162.53938551695828
# [0.92561724 0.02187332 0.11648356] -162.53938551693335
# [0.92561723 0.02187333 0.11648356] -162.5393855168556
# [0.92561723 0.02187332 0.11648357] -162.53938551674906
# reward process:
#       fun: 162.53938551695828
#      jac: array([-0.00167274, -0.00689125, -0.01403999])
#  message: 'Optimization terminated successfully'
#     nfev: 179
#      nit: 32
#     njev: 32
#   status: 0
#  success: True
#        x: array([0.92561723, 0.02187332, 0.11648356])
# time: 5948.5783812999725






# IPython 7.16.1 -- An enhanced Interactive Python.

# runfile('//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop/untitled1.py', wdir='//coe-fs.engr.tamu.edu/Grads/lusun8825/Desktop')
# theta 2:
#  [0.94916457 0.01221181]
# [0.94916457 0.01221181] -3656.5238063777106
# dynamic process:
#  [0.03916134 0.33333098 0.59039566 0.18078324 0.7575275  0.06068926]
# [9.738 0.3   1.3  ] -162.67922672298783
# [9.73800001 0.3        1.3       ] -162.67922671956498
# [9.738      0.30000001 1.3       ] -162.679226725636
# [9.738      0.3        1.30000001] -162.67922676301035
# [9.96770390e+00 1.00000000e-06 9.99999999e-07] -329.1368616011905
# [9.76097039 0.2700001  1.1700001 ] -162.88558050572064
# [9.74529881 0.29046758 1.25869272] -162.61824933719475
# [9.74529882 0.29046758 1.25869272] -162.6182493415484
# [9.74529881 0.29046759 1.25869272] -162.6182493339549
# [9.74529881 0.29046758 1.25869274] -162.61824934091413
# [9.42337954 0.53058514 1.14828365] -162.73027364651242
# [9.64742038 0.36347448 1.22512323] -162.59104689786562
# [9.64742039 0.36347448 1.22512323] -162.59104690027758
# [9.64742038 0.3634745  1.22512323] -162.5910469012165
# [9.64742038 0.36347448 1.22512324] -162.59104689865885
# [9.32544073 0.2479647  1.17243361] -162.54139408201354
# [9.32544074 0.2479647  1.17243361] -162.54139408289862
# [9.32544073 0.24796472 1.17243361] -162.54139408254738
# [9.32544073 0.2479647  1.17243363] -162.5413940798063
# [9.248917   0.21671666 1.16394291] -162.5394094895509
# [9.24891701 0.21671666 1.16394291] -162.53940948942315
# [9.248917   0.21671667 1.16394291] -162.53940948957015
# [9.248917   0.21671666 1.16394292] -162.53940948981452
# [9.25720901 0.21902725 1.16501568] -162.53938524312267
# [9.25720903 0.21902725 1.16501568] -162.53938524312716
# [9.25720901 0.21902726 1.16501568] -162.53938524311604
# [9.25720901 0.21902725 1.1650157 ] -162.53938524311116
# reward process:
#       fun: 162.53938524312267
#      jac: array([ 0.00030136, -0.00044441, -0.00077248])
#  message: 'Optimization terminated successfully'
#     nfev: 27
#      nit: 6
#     njev: 6
#   status: 0
#  success: True
#        x: array([9.25720901, 0.21902725, 1.16501568])
# time: 1863.5732078552246

