from sklearn import preprocessing
import pandas as pd
import numpy as np
import scipy.optimize as opt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import  shgo
from scipy.optimize import differential_evolution

#dat = pd.read_csv('group4.csv')
dat3 = pd.read_csv('group4_new_175.csv') # a dataframe has two columns, named "state" and "action".

import pickle

#with open("./data/fnn_{}_acc.txt".format(af), "rb") as fp:   #Pickling
#  fnn_time_tanh = pickle.load(fp)
#print(fnn_time_tanh)
#%%

class FullLikelihood(object):

    def __init__(self, data, Y, X, npars ,dim_x,dim_z,time, nbus,hide_state = True):

        """

        A statistics workbench used to evaluate the cost parameters underlying 

        a bus replacement pattern by a forward-looking agent.

        

        Takes:

            * Data: a Pandas dataframe, which contains:

                -Y: the name of the column containing the dummy exogenous 

                    variable (here, the choice)

                -X: the name of the column containing the endogenous variable 

                    (here, the state of the bus)



            * p: The state-transition vector of endogenous variable.

                    For instance, p = [0, 0.6, 0.4] means that the bus will 

                    transition to the next mileage state with probability 0.6, 

                    and to the second next mileage state with probability 0.4.



            * MF: A function passed as an argument, which is the functional 

                  form for the maintenance cost. This function must accept as

                  a first argument a state s, and as a second argument a vector

                  of parameters.

                  

            * npars: The number of parameters to evalutate (i.e. the number of 

                     parameters of the maintenance cost function, plus 1 for

                     the replacement cost)

            

            * time: The length of time horizon of the observations.

            

            * nbus: The number of buses.

        """        

        self.hide_state = hide_state

        self.endog = data.loc[:, Y].values

        self.exog = data.loc[:, X].values

        self.time = time

        self.nbus = nbus

        self.data = np.array(data)

        self.dim = dim_x 

        self.N = self.endog.shape[0]

        self.S = dim_z # Assumes that the true maximum number 

                                         # states is twice the maximum observed

                                         # state.

        self.D = [(i,j/(self.dim-1)) for i in range(self.S) for j in range(self.dim)]

        self.Dlength = self.S*self.dim

        #self.theta2 = theta2

        # Check that the stated number of parameters correspond to the

        # specifications of the maintenance cost function. 

        

              

        # To speed up computations and avoid loops when computing the log 

        # likelihood, we create a few useful matrices here:

        length = self.Dlength

        

        

        # A (SxS) matrix indicating the probability of a bus transitioning

        # from a state s to a state s' (used to compute maintenance utility)

                # A second (SxS) matrix which regenerates the bus' state to 0 with

        # certainty (used to compute the replacement utility)
        #print(np.zeros((self.dim-1, length)).shape)

        self.regen_mat = np.vstack((np.zeros((self.dim-1, length)),np.ones((1,length)),np.zeros((length-self.dim, length))))

        

        # A (2xN) matrix indicating with a dummy the decision taken by the agent

        # for each time/bus observation (replace or maintain)

        self.dec_mat = np.vstack(((1-self.endog), self.endog))

        

    def trans(self, theta2,p0,p1):

        """

        This function generate the transition matrix

            

            * theta2: the true state transition probability

            

            * p0: the transition probability when s = 0

            

            * p1: the transition probability when s = 1

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

        #p_2 = np.array(([0,0],p_2)).max(0)

        self.P = P = np.array((p_0,p_1,p_2,p_3))

        #print(self.P)

        trans_mat = np.zeros((length, length))

        new_x0 = lambda x: np.dot(p_0*np.array([x,1-x]),theta2)/np.dot(p_0,[x,1-x])

        new_x1 = lambda x: np.dot(p_1*np.array([x,1-x]),theta2)/np.dot(p_1,[x,1-x])

        new_x2 = lambda x: np.dot(p_2*np.array([x,1-x]),theta2)/np.dot(p_2,[x,1-x])

        new_x3 = lambda x: np.dot(p_3*np.array([x,1-x]),theta2)/np.dot(p_3,[x,1-x])

        matrix0 = [new_x0(i/(dim-1)) for i in range(dim)]

        matrix1 = [new_x1(i/(dim-1)) for i in range(dim)]

        matrix2 = [new_x2(i/(dim-1)) for i in range(dim)]

        matrix3 = [new_x3(i/(dim-1)) for i in range(dim)]

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

                    #x_c = np.ceil((self.dim-1)*x_new)/(self.dim-1)

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

        

    def single_trans(self, p,gap):

        S = self.S

        trans_mat = np.zeros((S, S))

        for i in range(S):

            if i + gap < S-1:

                trans_mat[i+gap][i] = p

            elif i + gap == S-1:

                trans_mat[S-1][i] = p

            else:

                pass

        return trans_mat

        """

        self.trans_mat[S-1][S-1] = 0

        self.trans_mat[S-1][0] = 1

        """

    def belief(self,theta2,p0,p1, path):

        """

        This function return the belief x

        Takes:

            * theta2: the true state transition probability

            

            * p0: the transition probability when s = 0

            

            * p1: the transition probability when s = 1

            

            * A path of mileage and replacement\maintenance record data

        """

        length = self.time

        x = [1]

        p_0 = np.array([p0[0],p1[0]])

        p_1 = np.array([p0[1],p1[1]])

        p_2 = np.array([p0[2],p1[2]])

        p_3 = 1-p_0-p_1-p_2
        #print(length,path)

        for i in range(length-1):

            if path[i][1] == 1:

                x_new = 1

                x.append(x_new)

            else:

                #x_f = np.floor((self.dim-1)*x[-1])/(self.dim-1)

                #x_c = np.ceil((self.dim-1)*x[-1])/(self.dim-1)

                #coe = (x[-1]-x_f)*(self.dim - 1)

                #x_cur = np.array([x_c*coe+x_f*(1-coe), coe*(1-x_c)+ (1-coe)*(1-x_f)])

                x_cur= np.array([x[-1],1-x[-1]])

                gap = path[i+1][0]-path[i][0]

                if gap == 0:

                    v1 = np.dot(p_0,x_cur)

                    v2 = np.dot(p_0,np.diag(x_cur))

                    v3 = np.dot(v2,theta2)

                    x_new = v3/v1

                    x.append(x_new)

                elif gap == 1:

                    v1 = np.dot(p_1,x_cur)

                    v2 = np.dot(p_1,np.diag(x_cur))

                    v3 = np.dot(v2,theta2)

                    x_new = v3/v1

                    x.append(x_new)

                elif gap == 2:

                    v1 = np.dot(p_2,x_cur)

                    v2 = np.dot(p_2,np.diag(x_cur))

                    v3 = np.dot(v2,theta2)

                    x_new = v3/v1

                    x.append(x_new)

                else:

                    v1 = np.dot(p_3,x_cur)

                    v2 = np.dot(p_3,np.diag(x_cur))

                    v3 = np.dot(v2,theta2)

                    x_new = v3/v1

                    x.append(x_new)

        return x

    """

    def v_value(self,params,theta2,p0,p1 ,beta=0.9999):

        S = self.S

        dim = self.dim

        p_0 = np.array([p0[0],p1[0]])

        p_1 = np.array([p0[1],p1[1]])

        p_2 = 1-p_0-p_1

        rc = params[0]

        thetas = np.array(params[1:])

        maintain = lambda x,s: np.dot(0.001*s*thetas,[x,1-x])

        self.q_0 = [maintain(j/(dim-1),0) for j in range(0,dim)]

        for i in range(S-1):

            q = [maintain(j/(dim-1),i+1) for j in range(0,dim)]

            self.q_0 = np.vstack((self.q_0,q))

        reward_0 = self.q_0

        reward_1 = self.q_1 = np.zeros((S,dim))+rc

        m = np.array((self.q_0,self.q_1)).min(0)

        EV = np.log(np.exp(-self.q_0+m)+np.exp(-self.q_1+m))-m+0.5772

        for i in range(1000):

            new_x0 = lambda x: np.floor((self.dim-1)*p_0[0]*x*theta2/np.dot(p_0,[x,1-x]))/(self.dim-1)

            new_x1 = lambda x: np.floor((self.dim-1)*p_1[0]*x*theta2/np.dot(p_1,[x,1-x]))/(self.dim-1)

            new_x2 = lambda x: np.floor((self.dim-1)*p_2[0]*x*theta2/np.dot(p_2,[x,1-x]))/(self.dim-1)

            self.q_0 = reward_0 + beta*[]

            

    """

    def myopic_costs(self, params):

        length = self.Dlength

        D = self.D

        """

        This function computes the myopic expected cost associated with each 

        decision for each state.

        

        Takes:

            * A vector params, to be supplied to the maintenance cost function 

              MF. The first element of the vector is the replacement cost rc.

              



        Returns:

            * A (Sx2) array containing the maintenance and replacement costs 

              for the S possible states of the bus

        """

        rc = params[0]

        thetas = np.array(params[1:])

        maint_cost = np.array([np.dot(0.001*D[d][0]*thetas,[D[d][1],1-D[d][1]]) for d in range(0, length)])

        repl_cost = [rc for d in range(0, length)]

        #maint_cost[S-1] = 1e+10

        return  np.vstack((maint_cost, repl_cost)).T

    

    def fl_costs(self, params,theta2,p0,p1 ,beta=0.9999):

        """

        Compute the non-myopic expected value of the agent for each possible 

        decision and each possible state of the bus, conditional on a vector of 

        parameters and on the maintenance cost function specified at the 

        initialization of the DynamicUtility model.



        Iterates until the difference in the previously obtained expected value 

        and the new expected value is smaller than a constant.

        

        Takes:

            * A vector params for the cost function

            * theta2: hiden state transition probability

            * p0: the transition probability when s = 0

            * p1: the transition probability when s = 1

            * A discount factor beta (optional)

            * A convergence threshold (optional)

            * A boolean argument to suppress the output (optional)



        Returns:

            * An (Sx2) array of forward-looking costs associated with each

              state and each decision.

        """

        # Initialization of the contraction mapping

        self.EV_myopic = EV_new =self.myopic_costs(params)

        self.trans_mat = self.trans(theta2,p0,p1)

        # Contraction mapping Loop

        for i in range(1000):

            length = self.Dlength

            #D = self.D

            #pchoice = self.choice_prob(EV)

            #ecost = (pchoice*EV).sum(1)-0.5772

            EV = EV_new

            m = EV.min(1).reshape(length, -1)

            ecost = np.log(np.exp(-EV+m).sum(1) )-m.T+0.5772

            

            #ecost = lambda x: np.log(np.exp(-EV(x)).sum(1)) + 0.5772

            futil_maint = np.dot(ecost, self.trans_mat)

            futil_repl = np.dot(ecost, self.regen_mat)

            futil = np.vstack((futil_maint, futil_repl)).T

            EV_new = self.EV_myopic - beta*futil



        # Output:

        """

        if not suppr_output:

            if achieved:

                print("Convergence achieved in {} iterations".format(k))

            else:

                print("CM could not converge! Mean difference = {:.4f}".format(

                                                            (EV_new-EV).mean())

                                                                              )

    """

                

        self.EV_FL = EV_new

        return EV_new



    def choice_prob(self,rho ,cost_array):

        """

        Returns the probability of each choice for each observed state, 

        conditional on an array of state/decision costs (generated by the 

        myopic or forward-looking cost functions)

        """

        cost_array = cost_array/rho

        cost = cost_array - cost_array.min(1).reshape(self.Dlength, -1)

        util = np.exp(-cost)

        pchoice = util/(np.sum(util, 1).reshape(self.Dlength, -1))

        return pchoice

        

    def loglike2(self,parameters):

        theta2 = self.theta2

        p0 = parameters[0:3]

        p1 = parameters[3:6]

        prob0 = np.array([p0[0],p1[0]])

        #prob0 = np.around(prob0,3)

        prob1 = np.array([p0[1],p1[1]])

        #prob1 = np.around(prob1,3)

        prob2 = np.array([p0[2],p1[2]])

        prob3 = 1 - prob0 -prob1 - prob2

        mileage = self.exog

        if self.hide_state == True:

            self.beli = []

            for i in range(self.nbus):
                #print(i,len(self.beli))

                self.beli = self.beli + self.belief(theta2,p0,p1,self.data[i*self.time:(i+1)*self.time])

        else:

            self.beli = [1 for s in range(0,self.N)]
        #print('end')

        logprob = 0

        for i in range(self.N-1):

            x_cur = [self.beli[i], 1-self.beli[i]]

            if mileage[i+1] - mileage[i] == 0:

                logprob += np.log(np.dot(prob0,x_cur))

            elif mileage[i+1] - mileage[i] == 1:

                logprob += np.log(np.dot(prob1,x_cur))

            elif mileage[i+1] - mileage[i] == 2:

                logprob += np.log(np.dot(prob2,x_cur))

            elif mileage[i+1] - mileage[i] == 3:

                logprob += np.log(np.dot(prob3,x_cur))
        #print('end')

        return -logprob

    def loglike(self, parameters):

        """

        The log-likelihood of the Dynamic model is estimated in several steps.

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

        #theta2 = self.fitted_full.x

        p0 = self.fitted2.x[0:3]

        p1 = self.fitted2.x[3:6]

        theta2 = self.theta2

        #p0 = self.p0

        #p1 = self.p1

        #prob2 = np.array(([0,0],prob2)).max(0)

        util = self.fl_costs(params,theta2,p0,p1) 

        self.pchoice = pchoice =self.choice_prob(rho,util)

        mileage = self.exog

        self.beli = []
        #print(self.data.shape())

        for i in range(self.nbus):
            #print(i)
            #print(self.data[i*self.time:(i+1)*self.time])
            #print(self.belief(theta2,p0,p1,self.data[i*self.time:(i+1)*self.time]))

            self.beli = self.beli + self.belief(theta2,p0,p1,self.data[i*self.time:(i+1)*self.time])

        # A (SxN) matrix indicating the state of each observation

        b_f = np.floor((self.dim-1)*np.array(self.beli))/(self.dim-1)

        self.b_coe = (self.beli-b_f)*(self.dim-1)

        self.state_mat_f = np.array([[(1-self.b_coe[i])*(mileage[i]==j and np.floor(self.beli[i]) == m) for i in range(self.N)] 

                                                    for (j,m) in self.D])

        self.state_mat_c = np.array([[self.b_coe[i]*(mileage[i]==j and np.ceil(self.beli[i]) == m) for i in range(self.N)] 

                                                    for (j,m) in self.D])

        self.state_mat = self.state_mat_f+self.state_mat_c 

        logprob = np.log(np.dot(pchoice.T, self.state_mat))

        logprob = np.sum(self.dec_mat*logprob)

        print(parameters,logprob)

        #logprob = np.log(np.dot(pchoice.T, self.state_mat))

        #return -np.sum(self.dec_mat*logprob)-(count_0 * np.log(p1) + count_1 * np.log(p2) + count_2 * np.log(p3))

        return -logprob

    

    def loglike_full(self,theta2):

        self.theta2 = theta2

        self.fit_likelihood2()

        loglike = -self.fitted2.fun 
        print(theta2,loglike)

        return -loglike

    def fit_likelihood(self, x0=None, bounds=None):

        """

        estimate the reward parameter

        """

        if bounds == None:

            bounds = [(1e-6, 100),(1e-6,100),(1e-6,100)]

            

        if x0 == None:

            x0 =  [10,0.01,0.01]

        cons= ({'type': 'ineq', 'fun': lambda x: x[2]-x[1] })

        #cons= ({'type': 'ineq', 'fun': lambda x: 1-x[4]-x[5]-1e-6 },{'type': 'ineq', 'fun': lambda x: 1-x[6]-x[7]-1e-6 })

        self.fitted = minimize(self.loglike, x0=x0,bounds = bounds, constraints= cons)

        #self.fitted = opt.fmin_l_bfgs_b(self.loglike, x0=x0, approx_grad=True, 

                                       # bounds=bounds)

        #self.fitted = differential_evolution(self.loglike, bounds)

    def fit_likelihood2(self):

        """

        estimate the dynamic

        """

        bounds = [(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),(0.001, 0.998),(0.001, 0.998)]

        x0 = [0.1,0.1,0.1,0.1,0.1,0.1]

        cons= ({'type': 'ineq', 'fun': lambda x: 1-x[0]-x[1]-x[2]-0.001 },{'type': 'ineq', 'fun': lambda x: 1-x[3]-x[4]-x[5]-0.001 },\

                {'type': 'ineq', 'fun': lambda x: -3*x[0]-2*x[1]-x[2]+3*x[3]+2*x[4]+x[5] })

        self.fitted2 = minimize(self.loglike2, x0=x0,bounds = bounds, constraints= cons)

    def fit(self):

        """

        estimate the belif parameter

        """

        bounds = [(0.9,0.99999),(0,0.09999)]

        #x0 = [0.1,0.1]

        #self.fitted_full = minimize(self.loglike_full, x0=x0,bounds = bounds)

        #self.fitted_full  = differential_evolution(self.loglike_full, bounds)

        #self.fitted_full = shgo(self.loglike_full, bounds)

        self.fitted_full = opt.brute(self.loglike_full, bounds,Ns=3)

        #self.theta2 = self.fitted_full.x

        self.theta2 = self.fitted_full

    

     #%%

"""

def lin_cost(s, params, x):

    try:

        theta1 = params

        return np.dot(s*theta1, [x,1-x])

    except ValueError:

        raise ValueError

        print("Wrong number of parameters specified: expected 2, got {}".format(len(params))

"""



#estimation = FullLikelihood(dat3,'action','state', npars=2,dim_x =51,dim_z = 175,time = 117, nbus = 37,hide_state = True)       


############################My Code###############################
import random


def KfoldData(Kfold,nbus = 37,time = 117,data = dat3,seed=0):
    data = np.array(dat3)
    Kfold = int(Kfold)
    
    index_bus = np.arange(nbus,dtype=int)
    random.seed(seed)
    random.shuffle(index_bus)
    
    ones_array = np.ones((1,time),dtype=int)
    arange_array = np.arange(time,dtype=int)
    
    data_sub = []
    nbus_sub = []
    
    for i in range(Kfold):
        index_tep = np.reshape(time * index_bus[i::Kfold].reshape((-1,1)) *\
                               ones_array+ arange_array,(1,-1))[0]
        #print(index_tep)
        numpy_data = np.delete(data,index_tep,axis=0)
        df = pd.DataFrame(data=numpy_data, columns=['state', 'action'])
        
        data_sub.append(df)
        nbus_sub.append(int(df.shape[0]/time))
    return data_sub,nbus_sub

Kfold = 5
time = 117
nbus = 37
seed = 1
data_sub,nbus_sub = KfoldData(Kfold,nbus,time ,data = dat3,seed=seed)

for j in range(Kfold-1):
    i = j+1
    data_record = [] #theta2 logSigma theta3 ccp [Rc,r1,r2]
    estimation = FullLikelihood(data_sub[i],'action','state', 
                                npars=2,dim_x =51,dim_z = 175,time = 117,
                                nbus = nbus_sub[i],hide_state = True)
    if j!=0:
        estimation.fit()
        data_record.append(estimation.fitted_full)
        print(estimation.fitted_full) #theta2
    else:
        estimation.loglike_full([0.95302141, 0.01012263])
        data_record.append([0.95302141, 0.01012263])
        print(data_record)
    
    estimation.fit_likelihood2()
    
    print(estimation.fitted2)     #theta3 and logSigma
    data_record.append(estimation.fitted2.fun)
    data_record.append(estimation.fitted2.x)
    
    estimation.fit_likelihood()
    
    print(estimation.fitted) #r Rc  and CCP
    data_record.append(estimation.fitted.fun)
    data_record.append(estimation.fitted.x)
    with open("./data/seed{}_{}Kfold_{}.txt".format(seed,Kfold,i+1), "wb") as fp:   #Pickling
        pickle.dump(data_record, fp)

#############################End##################################

"""

x0 = [10.2910,2.2,2.5,0.99,0,0.257,0.717,0.526,0.473]

a = estimation.loglike_full(x0)

#b = estimation.beli [10,0.01,0.02,0,0.4,0.55,0.4,0.55]

estimation.print_parameters()

#myopic = estimation.myopic_costs([10,1,1])

"""



