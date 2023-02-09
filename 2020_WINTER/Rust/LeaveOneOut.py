# # -*- coding: utf-8 -*-
# """
# Created on Sat Nov 14 00:47:19 2020
# 11:44
# @author: Sunlu
# """

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

class FullLikelihood(object):
    def __init__(self, data, Y, X, dim_x, dim_z, time, nbus, hide_state = True, disp_status = False):
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
     
    def fl_costs(self, params, theta2, p0, p1 ,beta=0.9999, threading = 0.45):
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
            ecost = np.log(np.exp(-EV+m).sum(1) )-m.T+0.5772
          
            futil_maint = np.dot(ecost, self.trans_mat)
            futil_repl = np.dot(ecost, self.regen_mat)
            futil = np.vstack((futil_maint, futil_repl)).T
            EV_new = self.EV_myopic - beta * futil

            i +=1
            VError = np.max(abs(EV-EV_new))

            if i>1000:
                break
            
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

        return pchoice

    # def loglike_full(self,theta2):
    #     """
    #     The log-likelihood of the Dynamic model(theta 2) is estimated.
    #     """
    #     self.theta2 = theta2
    #     self.fit_likelihood2()
        
    #     loglike = -self.fitted2.fun
        
    #     if self.disp_status:
    #         print(theta2,loglike)

    #     return -loglike
    def loglike2(self,parameters):
        #theta2 = self.theta2
        theta2 = parameters[6:]
        
    # def loglike2(self,parameters):
    #     """
    #     The log-likelihood of the Dynamic model(theta 3) is estimated.
    #     """


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
        print(parameters,logprob)
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
    

    # def fit(self):
    #     """
    #     estimate the belif parameter(theta 2)
    #     """
    #     # inorder to accelerate the computing speed
    #     bounds = [(0.9,0.99999),(0,0.09999)]
    #     self.fitted_full = opt.brute(self.loglike_full, bounds, Ns = 3) # To speed up, Ns=3
        
    #     self.theta2 = self.fitted_full

    def fit_likelihood2(self):
        """
        estimate the dynamic(theta 3)
        """
        # # bounds = [(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        # #           (0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        # #           (0,0.99999),(0,0.99999)]
        x0 = [0.03,0.3,0.5,0.1,0.7,0.06, 0.9, 0.1]
        cons= ({'type': 'ineq', 'fun': lambda x: x[0]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: x[1]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: x[2]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: x[3]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: x[4]-0.001},\
                {'type': 'ineq', 'fun': lambda x: x[5]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: x[6]},\
                {'type': 'ineq', 'fun': lambda x: x[7]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[0]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[1]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[2]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[3]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[4]},\
                {'type': 'ineq', 'fun': lambda x: 0.998-x[5]},\
                {'type': 'ineq', 'fun': lambda x: 0.99999-x[6]},\
                {'type': 'ineq', 'fun': lambda x: 0.99999-x[7]},\
                {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]-x[2]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: 1-x[3]-x[4]-x[5]-0.001 },\
                {'type': 'ineq', 'fun': lambda x: -3*x[0]-2*x[1]-x[2]+3*x[3]+2*x[4]+x[5] })
        # x0 = [0.1,0.1,0.1,0.1,0.1,0.1]
        # cons= ({'type': 'ineq', 'fun': lambda x: x[0]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[1]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[2]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[3]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: x[4]-0.001},\
        #        {'type': 'ineq', 'fun': lambda x: x[5]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[0]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[1]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[2]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[3]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[4]},\
        #        {'type': 'ineq', 'fun': lambda x: 0.998-x[5]},\
        #        {'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]-x[2]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: 1-x[3]-x[4]-x[5]-0.001 },\
        #        {'type': 'ineq', 'fun': lambda x: -3*x[0]-2*x[1]-x[2]+3*x[3]+2*x[4]+x[5] })
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
            x0 = [9,0.5,1]
        
        p0 = self.fitted2.x[0:3]
        p1 = self.fitted2.x[3:6]
        #theta2 = self.theta2
        theta2 = self.fitted2.x[6:8]
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
        
    def accuracy1(self,parameters):
        #theta2 = self.theta2
        theta2 = parameters[6:]

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
                logprob += np.dot(prob0,x_cur)

            elif mileage[i+1] - mileage[i] == 1:
                logprob += np.dot(prob1,x_cur)           

            elif mileage[i+1] - mileage[i] == 2:
                logprob += np.dot(prob2,x_cur)
                
            elif mileage[i+1] - mileage[i] == 3:
                logprob += np.dot(prob3,x_cur)
            elif self.endog[i]==1:
                logprob += 1

        #print(f'current likelihood: {logprob}')
        print(parameters,logprob)
        return logprob/self.time

    def accuracy2(self, parameters, p0, p1, theta2):
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
        

        sumprob = np.dot(pchoice.T, self.state_mat)
        sumprob = np.sum( self.dec_mat*sumprob)
        
        if self.disp_status:
            print(parameters,sumprob)

        #print(f'current likelihood:{parameters} {logprob}')

        return sumprob/self.time







# #%%


# ############################My Code###########################################
# import pickle
# import os
# #import ray

# #path = '/scratch/user/lusun8825/rust_hidden/'
# path = 'D:/Google Drive/2020_WINTER/Rust/'
# dat3 = pd.read_csv(path+'group4_new_175.csv') 

# nbus = 37
# time = 117

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
#     #np.random.seed(seed)
#     #samples = np.random.randint(0,nbus,nbus)
#     samples = [i for i in range(nbus)]
#     samples.remove(seed)
#     samples = np.array(samples)
#     state_new = np.zeros(time*(nbus-1))
#     action_new = np.zeros(time*(nbus-1))
#     print(samples)
#     for i,ss in enumerate(samples):
#         j = time*i
#         state_new[j:j+time] = data['state'][ss*time:(ss+1)*time]
#         action_new[j:j+time] = data['action'][ss*time:(ss+1)*time]
#     data = {'state':state_new,'action':action_new}
#     dat_new = pd.DataFrame(data=data)
#     #print(samples)
#     #print(dat_new)
#     return dat_new

# #@ray.remote

# def worker_task(sed,data=dat3,time = time,nbus=nbus,path = path):
#     warnings.filterwarnings("ignore")
#     #seed = current_work_index
#     #start = time.time()
#     #for i in range(10):
#     seed = 9-sed#sed*10 + i
#     print(seed)
#     #if seed <0:
#     #  data_new = data
#     #else:
#     #data_tol = []
#     #for seed in seed_tol:
#     #    #print(seed)
#     data_new = boostrap(seed,data=data,time=time,nbus=nbus)


#     #Record in form: seed theta2, logSigma, theta3, ccp, [Rc,r1,r2]
        
#     #print('\n\n{} seed,\n'.format(seed))
#     data_record = [] 
#     data_record.append(seed)

#     timeStart = tm.time()
    
#     #Initialize
#     estimation = FullLikelihood(data_new,'action','state', 
#                             dim_x =51,dim_z = 175,time = time, nbus = nbus-1, 
#                             hide_state = True, disp_status = True)
    
#     # #theta2
#     # estimation.fit()
#     # data_record.append(estimation.fitted_full)          #theta2
#     # print('theta 2:\n', estimation.fitted_full)
#     # data_record.append(estimation.fitted2.fun)          #logSigma
#     # data_record.append(estimation.fitted2.x)            #theta3
#     # print('dynamic process:\n',estimation.fitted2.x)

#     # data_record.append(estimation.fitted2.success)      #success1
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
#     print('time:',tm.time()-timeStart)
#     #Save record
#     if not os.path.exists(path+'data/seed_new'):
#         os.makedirs(path+'data/seed_new')
#     with open(path+'data/seed_new/cv_seed{}.txt'.format(seed), "wb") as fp:   #Pickling
#         pickle.dump(data_record, fp)
#     #data_tol.append(data_record)
#     return data_record#data_tol

# for i in range(0,4):
#     worker_task(i,data=dat3,time = time,nbus=nbus,path = path)
# #############################End#############################################
# # #if __name__ == "__main__":
# #ray.shutdown()
# #ray.init(num_cpus = 10)

# #worker_tasks = ray.get([worker_task.remote(x) for x in range(10)])

# #ray.shutdown()
# #5:30
############################My Code###########################################

import numpy as np
import pickle
import os

path_data =  'D:/Google Drive/2020_WINTER/Rust/data/seed_new/'

path = 'D:/Google Drive/2020_WINTER/Rust/'
dat3 = pd.read_csv(path+'group4_new_175.csv') 

nbus = 37
time = 117

def boostrap_test(seed,data=dat3,time=time,nbus=nbus):
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
    sample = seed
    state_new = np.zeros(time)
    action_new = np.zeros(time)
    print(sample)
    state_new = data['state'][sample*time:(sample+1)*time]
    action_new = data['action'][sample*time:(sample+1)*time]
    data = {'state':state_new,'action':action_new}
    dat_new = pd.DataFrame(data=data)
    
    return dat_new

#@ray.remote

def worker_task_test(sed,theta3,theta2,theta1,data=dat3,time = time,nbus=nbus,path = path):
    warnings.filterwarnings("ignore")
    seed = sed
    print(seed)
    data_new = boostrap_test(seed,data=data,time=time,nbus=nbus)


    #Record in form: seed theta2, logSigma, theta3, ccp, [Rc,r1,r2]
        
    #print('\n\n{} seed,\n'.format(seed))
    data_record = [] 
    data_record.append(seed)

    timeStart = tm.time()
    
    #Initialize
    estimation = FullLikelihood(data_new,'action','state', 
                            dim_x =51,dim_z = 175,time = time, nbus = 1, 
                            hide_state = True, disp_status = True)
    
        
    #theta3 and logSigma
    LLV1 = estimation.accuracy1(np.hstack([theta3,theta2]))
    LLV2 = estimation.accuracy2(theta1, theta3[0:3], theta3[3:6], theta2)
    
    return LLV1,LLV2#data_tol


#############################End#############################################
# #if __name__ == "__main__":
#ray.shutdown()
#ray.init(num_cpus = 10)

#worker_tasks = ray.get([worker_task.remote(x) for x in range(10)])

res = []
theta2 = np.zeros([37,2])
theta3 = np.zeros([37,6])
theta1 = np.zeros([37,3])
LLV = np.zeros([37,3]) 
for i in range(37):
    with open(path_data +'cv_seed{}.txt'.format(i),'rb' ) as fp:
        res.append(pickle.load(fp))
        theta2[i,:] = res[-1][1]
        theta3[i,:] = res[-1][3]
        theta1[i,:] = res[-1][6]
        LLV[i,0],LLV[i,1],LLV[i,2] = -res[-1][2],-res[-1][5],-res[-1][2]-res[-1][5]

acc = np.zeros([37,3])
for i in range(37):
    
    acc[i,0],acc[i,1] = worker_task_test(i,theta3[i],theta2[i],theta1[i])
    acc[i,2] = (acc[i,0] + acc[i,1])/2



with open(path+'data/seed_new/cv_acc.txt', "wb") as fp:   #Pickling
    pickle.dump(acc, fp)


# (array([0.50584042, 0.98486561, 0.74535302]),
#  array([0.0238048 , 0.0045303 , 0.01260409]))
# array([0.43322201, 0.97635836, 0.70479018]),
#  array([0.01983952, 0.00454503, 0.01036692]))