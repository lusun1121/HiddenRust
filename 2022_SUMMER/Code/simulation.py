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

from tqdm import tqdm
warnings.filterwarnings("ignore")
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
    def __init__(self, data, Y, X,belief0, dim_x, dim_z,time, nbus, alpha,hide_state = True, disp_status = False):
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
        self.alpha = alpha
        #self.endog = data.loc[:, Y].values
        self.exog = data.loc[:, X].values
        self.dim = dim_x     
        self.S = dim_z 
        self.time = time
        self.nbus = nbus
        self.hide_state = hide_state
        self.disp_status = disp_status
        self.belief0 = belief0
        # To speed up computations and avoid loops when computing the log 
        # likelihood, we create a few useful matrices here:
        self.N = data.loc[:, Y].values.shape[0]
        self.Dlength = self.S*self.dim
        #length = self.Dlength
        
        # A (length x 2) matrix indicating the blief of a bus at observation 
        # state i 
        self.D = [(i,j/(self.dim-1)) for i in range(self.S) 
                                     for j in range(self.dim)]
        

        
        # A (length x length) matrix indicating the probability of a bus 
        # transitioning from a state z to a state z' with centain belif  
        # self.regen_mat = np.vstack((np.zeros((self.dim-1, self.Dlength)),
        #                             np.ones((1,self.Dlength)),
        #                             np.zeros((self.Dlength-self.dim, self.Dlength))))
        
        # A (2xN) matrix indicating with a dummy the decision taken by the 
        # agent for each time/bus observation (replace or maintain)        
        #self.dec_mat = np.vstack(((1-data.loc[:, Y].values), data.loc[:, Y].values))
               
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
        
        #self.P = P = np.array((p_0,p_1,p_2,p_3))
        P = np.array((p_0,p_1,p_2,p_3))
        trans_mat = np.zeros((length, length))
        new_x0 = lambda x: np.dot(p_0*np.array([x,1-x]),theta2) / np.dot(p_0,[x,1-x])
        new_x1 = lambda x: np.dot(p_1*np.array([x,1-x]),theta2) / np.dot(p_1,[x,1-x])
        new_x2 = lambda x: np.dot(p_2*np.array([x,1-x]),theta2) / np.dot(p_2,[x,1-x])
        new_x3 = lambda x: np.dot(p_3*np.array([x,1-x]),theta2) / np.dot(p_3,[x,1-x])
        
        matrix0 = [new_x0(i / (dim-1)) for i in range(dim)]
        matrix1 = [new_x1(i / (dim-1)) for i in range(dim)]
        matrix2 = [new_x2(i / (dim-1)) for i in range(dim)]
        matrix3 = [new_x3(i / (dim-1)) for i in range(dim)]
        
        #self.matrix = x_matrix = np.vstack((matrix0,matrix1,matrix2,matrix3))
        x_matrix = np.vstack((matrix0,matrix1,matrix2,matrix3))
        
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
    
    def belief(self,theta2,p0,p1, path ,x,data_gene_status = False):
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
        if data_gene_status:
            #path = [[zold,action],[znew,0]]
            length = 2
            #x = [path[0][2]] #return(xold,xnew)           
        
        else:
            length = self.time
            #x = [1]
        x = [x]
        
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
    
    def simulation(self, dim_x0,SampleSize, TimePeriod, reward = [9.243,0.226,1.164],theta2 = [0.949,0.012],\
                    p0 = [0.039,0.333,0.590],p1=[0.181,0.757,0.061],rand_status = False):
        reward = [ i*self.alpha for i in reward]
        print(reward)
        condition = np.zeros([SampleSize,TimePeriod],dtype = int) #hidden state: 0,1
        state = np.zeros([SampleSize,TimePeriod],dtype = int ) #mileage: 175:0,...,174
        x = np.ones([SampleSize,TimePeriod])#belief: s = 0 prob # 删除
        if rand_status: # s0 z0 x0
            x[:,0] = np.kron(np.linspace(0, 1,num= dim_x0),np.ones(int(SampleSize/dim_x0)))
        
            #x[:,0] = np.kron(np.random.uniform(size = dim_x0),np.ones(int(SampleSize/dim_x0)))
            state[:,0] = np.random.choice(np.arange(110,dtype= int),size = SampleSize,p=np.ones(110)/110)#uniform(0,20)
            pis =0.9509143407122231#np.random.uniform()#0.5#
            condition[:,0] = np.random.choice([0,1],size = SampleSize, p = [pis,1-pis])#0000
        else:
            pis = 1
        action = np.zeros([SampleSize,TimePeriod],dtype = int)
        rho = 1
        print("generate Q")
        Q = self.fl_costs( reward, theta2, p0, p1 ) # Q function Q(x,a) --> interpolation 
        pchoice =self.choice_prob(rho,Q) # z = 0, x= 0, 0.1, 0.2,....,1, z = 1, x = 0,...,1 pi
        print("generate dataset")
        for nbus in tqdm(range(SampleSize)):
            for t in range(TimePeriod-1):
                path = np.zeros([2,2])
                path[0,0] = state[nbus,t]
                #path[0,2] = x[nbus,t] 
                
                fl =  int(np.floor(x[nbus,t]*(self.dim-1)))
                cl =  int(np.ceil(x[nbus,t]*(self.dim-1)))
                coef = x[nbus,t] * (self.dim-1)-fl
                pi0 = pchoice[state[nbus,t] * self.dim + fl,0] * (1-coef) +\
                    pchoice[state[nbus,t] * self.dim + cl,0] *coef
                    
                action[nbus,t] = np.random.choice([0,1],p=[pi0,1-pi0])
                path[0,1] = action[nbus,t]
                
                if action[nbus,t]==1:
                    state[nbus,t+1] = 0
                    condition[nbus,t+1] = 0
                    path[1,0] = state[nbus,t+1]
                else:
                    if condition[nbus,t] == 0:
                        state[nbus,t+1] = state[nbus,t] + np.random.choice([0,1,2,3],p=[p0[0],p0[1],p0[2],1-np.sum(p0)])
                        if state[nbus,t+1]>self.S-1:
                            state[nbus,t+1] = self.S-1
                            
                        condition[nbus,t+1] = np.random.choice([0,1],p=[theta2[0],1-theta2[0]])
                        path[1,0] = state[nbus,t+1]  
                    else:
                        state[nbus,t+1] = state[nbus,t] + np.random.choice([0,1,2,3],p=[p1[0],p1[1],p1[2],1-np.sum(p1)])
                        if state[nbus,t+1]>self.S-1:
                            state[nbus,t+1] = self.S-1
                        condition[nbus,t+1] = np.random.choice([0,1],p=[theta2[1],1-theta2[1]])
                        path[1,0] = state[nbus,t+1]
                x[nbus,t+1] = self.belief(theta2,p0,p1, path,x[nbus,t] ,data_gene_status = True)[-1]
                
            fl =  int(np.floor(x[nbus,TimePeriod-1]*(self.dim-1)))
            cl =  int(np.ceil(x[nbus,TimePeriod-1]*(self.dim-1)))
            coef = x[nbus,TimePeriod-1] * (self.dim-1)-fl
            pi0 = pchoice[state[nbus,TimePeriod-1] * self.dim + fl,0] * (1-coef) +\
                pchoice[state[nbus,TimePeriod-1] * self.dim + cl,0] *coef
                
            action[nbus,TimePeriod-1] = np.random.choice([0,1],p=[pi0,1-pi0])        
          
        state_new = np.reshape(state,[-1])
        action_new = np.reshape(action,[-1])
        data = {'state':state_new,'action':action_new}
        dat_new = pd.DataFrame(data=data)
        
        return dat_new,{'state':state,'action':action,'hidden':condition,'belief':x,'pis':pis}
    
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
     
    def fl_costs(self, params, theta2, p0, p1 ,beta=0.9999, threading = 0.45):#beta=0.9, threading = 1e-3):#beta=0.9999, threading = 3e-3):#
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
        #threading = threading*self.alpha
        # Initialization of the contraction mapping
        #self.EV_myopic =self.myopic_costs(params)
        params = np.array(params)/self.alpha
        EV_myopic =self.myopic_costs(params)
        
        EV_new = EV_myopic

        EV = np.zeros(EV_new.shape)        
        #VError = np.max(abs(EV-EV_new))
        VError = np.max(abs(EV-EV_new))/np.max(np.abs(EV))

        #print('Initial Error:',VError,np.max(abs(EV_new)))

        #self.trans_mat = self.trans(theta2,p0,p1)
        trans_mat = self.trans(theta2,p0,p1)
        # Contraction mapping Loop
        i = 0
        length = self.Dlength
        
        while(VError>threading):         

            EV = EV_new
            m = EV.min(1).reshape(length, -1)
            #ecost = np.log(np.exp(-EV+m).sum(1) )-m.T+0.5772
            ecost = (np.log(np.exp((-EV+m)).sum(1) )-m.T+0.5772)
          
            futil_maint = np.dot(ecost, trans_mat)
            futil_repl = np.dot(ecost, np.vstack((np.zeros((self.dim-1, self.Dlength)),
                                    np.ones((1,self.Dlength)),
                                    np.zeros((self.Dlength-self.dim, self.Dlength)))))
            futil = np.vstack((futil_maint, futil_repl)).T
            EV_new = EV_myopic - beta * futil

            i +=1
            VError = np.max(abs(EV-EV_new))
            #print(i,'th Error:',VError,VError/np.max(abs(EV)),np.max(abs(EV_new)))

            #VError = np.max(abs(EV-EV_new))/np.max(np.abs(EV))
            #print(i,'th Error:',VError,np.max(abs(EV-EV_new)),np.max(abs(EV_new)))

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
        #util = np.exp(-cost)
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
        if len(parameters)==8:
            theta2 = parameters[6:8]
        else:
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
            beli = []
            
            for i in range(self.nbus):
                beli = beli + self.belief(theta2,p0,p1,
                                self.data[i*self.time:(i+1)*self.time],self.belief0[i])
        else:
            beli = [1 for s in range(0,self.N)]
        
        logprob = 0
        for i in range(self.N-1):
            x_cur = [beli[i], 1-beli[i]]

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

        if self.disp_status:
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
        # A (length x 1) matrix        
         

        beli = []
        #self.beli = []

        for i in range(self.nbus):
            beli = beli + self.belief(theta2,p0,p1,
                                        self.data[i*self.time:(i+1)*self.time],self.belief0[i])


        # bool_to_int_f = np.array(((np.array((np.ones((self.Dlength,1)) * self.exog.reshape((1,self.N))) == (np.array(self.D).T[0].reshape((self.Dlength,1)) * np.ones((1,self.N))), dtype=int)) + \
        #                           np.array((np.ones((self.Dlength,1)) * np.floor((np.array(self.beli).reshape((1,self.N)))*(self.dim-1)) / (self.dim-1)) == (np.array(self.D).T[1].reshape((self.Dlength,1)) * np.ones((1,self.N))), dtype=int)) == 2, dtype=int)
        # bool_to_int_c = np.array(((np.array((np.ones((self.Dlength,1)) * self.exog.reshape((1,self.N))) == (np.array(self.D).T[0].reshape((self.Dlength,1)) * np.ones((1,self.N))), dtype=int)) +\
        #                           np.array((np.ones((self.Dlength,1)) * np.ceil((np.array(self.beli).reshape((1,self.N))) * (self.dim-1)) / (self.dim-1)) == (np.array(self.D).T[1].reshape((self.Dlength,1)) * np.ones((1,self.N))), dtype=int)) == 2, dtype=int)

        # #self.state_mat =
        
        # self.D = [(i,j/(self.dim-1)) for i in range(self.S) 
        #                              for j in range(self.dim)]
        bool_floor = np.int_(np.floor(np.array(beli)*(self.dim-1)))
        bool_ceil = np.int_(np.ceil(np.array(beli)*(self.dim-1)))
        bool_non0 = np.where((bool_floor-bool_ceil)!=0)
        coef_floor = np.zeros(bool_floor.shape)
        coef_floor[bool_non0] = (np.array(beli)*(self.dim-1)-bool_ceil)[bool_non0]/(bool_floor-bool_ceil)[bool_non0]
        
        choiceProb = self.choice_prob(rho,self.fl_costs(params,theta2,p0,p1)) #states,actions
        action = np.int_(self.data[:,1])
        statef =  np.int_(self.exog*self.dim+bool_floor)
        statec = np.int_(self.exog*self.dim+bool_ceil)
        logprob = coef_floor * choiceProb[statef,action] + (1-coef_floor)* choiceProb[statec,action] 
        
        logprob = np.sum(np.log(logprob))
        
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
    #                                     self.data[i*self.time:(i+1)*self.time],self.belief0[i])

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
                
        
    #     #self.pchoice = pchoice =self.choice_prob(rho,util) #



        
    #     # if self.disp_status:
    #     #     print(parameters,logprob)

    #     # print(f'current likelihood:{parameters} {logprob}')

    #     return parameters
    
    # def fit(self):
    #     """
    #     estimate the belif parameter(theta 2)
    #     """
    #     # inorder to accelerate the computing speed
    #     bounds = [(0,0.99999),(0,0.99999)]
    #     self.fitted_full = opt.brute(self.loglike_full, bounds, Ns = 3) # To speed up, Ns=3
        
    #     self.theta2 = self.fitted_full
    
    
    def fit_likelihood2(self):
        """
        estimate the dynamic(theta 3)
        """
        # bounds = [(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        #           (0.001, 0.998),(0.001, 0.998),(0.001, 0.998),
        #           (0,0.99999),(0,0.99999)]
        #x0 = [0.1,0.1,0.1,0.1,0.1,0.1, 0.1, 0.1]
        x0 = [0.039,0.333,0.590,0.181,0.757,0.061,0.949,0.012]
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
            constraints= cons,
            options={'disp': True})
        if len(x0)==8:
            self.theta2 = self.fitted2.x[6:8]
        
    def fit_likelihood(self, x0=None, bounds=None,p0 =None,p1=None ):
        """
        estimate the reward parameter
        """
        if bounds == None:
            bounds = [(1e-6, 100),(1e-6,100),(1e-6,100)]

        if x0 == None:
            #x0 = [10,1,1,]
            #x0 =  [9.36, 0.94, 0.94] # To speed up, choose a closer initial point
            #x0 = [0.1, 0.1, 0.1]
            x0 = np.array([9.243,0.226,1.164])*self.alpha#[9,0.5,1]
        if np.sum(p0==None) and np.sum(p1==None):
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
        
        # bounds = [(1e-6,15),(1e-6 ,2),(1e-6 ,2)]
        # func = lambda parameters: self. loglike(parameters, p0, p1, theta2)
        # self.fitted = opt.brute(func, bounds, Ns = 3) # To speed up, Ns=3
        



#%%
import pickle
import os
if __name__ == "__main__":
    
    print('Reminder1: file group4_new_175.csv should be in the same file directory')
    print('Reminder2: the code will take almost 4 hours')
    
    # a dataframe has two columns, named "state" and "action".    
    
    nbus = 3000
    time = 100
    dim_x = 100
    
    print('->Date generation')
    
    dat3 = pd.read_csv('group4_new_175.csv') 
    estimation1 = FullLikelihood(dat3,'action','state', np.ones(37),
                            dim_x =51,dim_z = 175,time = 117, nbus = 37, alpha =0.001,
                            hide_state = True, disp_status = False)
    res = estimation1.simulation(dim_x,nbus,time,rand_status = True)
    
    data_record = [] 
    data_record.append(res)
    
    # with open('data/sample1500.txt', "rb") as fp:   #Pickling
    #     res = pickle.load(fp)    
        
    # # datagenerate1 = FullLikelihood(res['res'][0],'action','state', res['res'][1]['belief'][:,0],
    # #                         dim_x =51,dim_z = 175,time = time, nbus = nbus, 
    # #                         hide_state = True, disp_status = False)
    # # res_add = datagenerate1.simulation(15,nbus,time,rand_status = True)
    # # datagenerate1.theta2 = [0.949,0.012]
    # # loglike1 = datagenerate1.loglike2(parameters=[0.039,0.333,0.590,0.181,0.757,0.061])
    # # loglike2 = datagenerate1.loglike(parameters=[9.243,0.226,1.164],\
    # #                 p0 = [0.039,0.333,0.590],p1=[0.181,0.757,0.061],theta2 = [0.949,0.012])
    # # print('true loglike1:' ,loglike1)
    # # print('true loglike2', loglike2)
    # data_record.append([0.949,0.012])
    # # data_record.append(loglike1)
    # data_record.append([0.039,0.333,0.590,0.181,0.757,0.061])
    # # data_record.append(loglike2)
    # data_record.append([9.243,0.226,1.164])
    

    #     #Record in form: 
    # #       theta2, logSigma, theta3, success1, ccp, [Rc,r1,r2], success2
    
    alpha_full = np.array([1])
    for alpha in alpha_full:
        print('->Estimation1 x0 known, alpha = ', alpha)
        timeStart = tm.time()
        
        #Initialize
        estimation1 = FullLikelihood(res[0].head(nbus*time),'action','state', res[1]['belief'][0:nbus*time,0],
                                dim_x =51,dim_z = 175,time = time, nbus = nbus, alpha =alpha,
                                hide_state = True, disp_status = True)
        estimation1.fit_likelihood2()
        data_record.append(estimation1.fitted2)
        print(estimation1.fitted2)
        estimation1.fit_likelihood()
        data_record.append(estimation1.fitted)
        print(estimation1.fitted)
    
    print(data_record)
    with open('sample_'+str(dim_x)+'_'+str(nbus)+'_'+str(time)+'_beta09999.txt', "wb") as fp:   #Pickling
        res = pickle.dump(data_record,fp)   
#     # # # #Record in form: 
#     # # # #       theta2, logSigma, theta3, success1, ccp, [Rc,r1,r2], success2
#     # # # data_record = [] 
#     # # #theta2
#     # # estimation1.fit()
#     # # data_record.append(estimation1.fitted_full)          #theta2
#     # # print('theta 2:\n', estimation1.fitted_full)
#     # # data_record.append(estimation1.fitted2.fun)          #logSigma
#     # # data_record.append(estimation1.fitted2.x)            #theta3
#     # # print('dynamic process:\n',estimation1.fitted2.x)

#     # # data_record.append(estimation1.fitted2.success)      #success1
#     # # #estimation.loglike_full([0.94946915,0.01199934])
    
    # # #[Rc,r1,r2] and ccp
    # data_record.append(res['data'][0])
    # estimation1.theta2 = res['data'][0].x[6:8]
    # estimation1.fit_likelihood(p0 = res['data'][0].x[0:3],p1=res['data'][0].x[3:6])
    # data_record.append(estimation1.fitted)
    
    # # data_record.append(estimation1.fitted.fun)           #ccp
    # # data_record.append(estimation1.fitted.x)             #[Rc,r1,r2]
    # # data_record.append(estimation1.fitted.success)       #success2
    # print('reward process:\n',estimation1.fitted)#2 minut        
    print('time:',tm.time()-timeStart)
    
#     print('->Estimation2 x0 unkonwn')
# #     timeStart = tm.time()
    
#     #Initialize
#     #rand_belief = res['randBelief']
#     rand_belief = np.random.uniform(size=nbus)
#     estimation1 = FullLikelihood(res[0].head(nbus*time),'action','state', rand_belief,
#                             dim_x =51,dim_z = 175,time = time, nbus = nbus, 
#                             hide_state = True, disp_status = True)
#     estimation1.fit_likelihood2()
#     data_record.append(estimation1.fitted2)
#     print(estimation1.fitted2)
#     estimation1.fit_likelihood()
#     data_record.append(estimation1.fitted)
#     print(estimation1.fitted)
#     haha = {'res':res,'data':data_record,'rand_belief':rand_belief}

#     # data_record.append(estimation2.fit_likelihood2())
#     # print(estimation2.fitted2)
    
#     # #Record in form: 
#     # #       theta2, logSigma, theta3, success1, ccp, [Rc,r1,r2], success2
    
#     # #theta2
#     # estimation2.fit()
#     # data_record.append(estimation2.fitted_full)          #theta2
#     # print('theta 2:\n', estimation2.fitted_full)
#     # data_record.append(estimation2.fitted2.fun)          #logSigma
#     # data_record.append(estimation2.fitted2.x)            #theta3
#     # print('dynamic process:\n',estimation2.fitted2.x)

#     # data_record.append(estimation2.fitted2.success)      #success1
#     # #estimation.loglike_full([0.94946915,0.01199934])
    
#     #[Rc,r1,r2] and ccp
#     estimation2.theta2 = res['unknown'].x[6:8]
#     estimation2.fit_likelihood(p0=res['unknown'].x[0:3],p1=res['unknown'].x[3:6])
#     data_record.append(estimation2.fitted.fun)           #ccp
#     data_record.append(estimation2.fitted.x)             #[Rc,r1,r2]
#     data_record.append(estimation2.fitted.success)       #success2
#     print('reward process:\n',estimation2.fitted)#2 minut        
#     print('time:',tm.time()-timeStart)
    
#     # # haha = {'res':res,'data':data_record,'rand':rand_belief}
#     # #
#     # haha = {'res':res,'data':data_record}
#     # if not os.path.exists('data/'):
#     #     os.makedirs('data/')
#     # with open('data/fix_001.txt', "wb") as fp:   #Pickling
#     #     pickle.dump(haha, fp)
    
#     #estimation.policy_gradient([9.38559752672684, 0.7182964136458119, 1.11155388413172])

# # with open('data/rand.txt',"rb") as fp:
# #     haha = pickle.load(fp)
    

# # res = haha['res']
# # data_record = haha['data']
# # rand_belief = haha['rand']
# # time = 117
# # nbus = 100
# # estimation2 = FullLikelihood(res[0],'action','state', rand_belief,
# #                         dim_x =51,dim_z = 175,time = time, nbus = nbus, 
# #                         hide_state = True, disp_status = True)
# # estimation2.theta2 = data_record[-4]

# # #[Rc,r1,r2] and ccp
# # estimation2.fit_likelihood(p0=data_record[-2][0:3],p1=data_record[-2][3:6])
# # data_record.append(estimation2.fitted.fun)           #ccp
# # data_record.append(estimation2.fitted.x)             #[Rc,r1,r2]
# # data_record.append(estimation2.fitted.success)       #success2
# # print('reward process:\n',estimation2.fitted)#2 minut