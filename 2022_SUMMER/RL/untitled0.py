# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:16:16 2020

@author: lusun
"""
import numpy as np
def funPrint(x):
  print(len(x))
  for i in range(len(x)):
    print(x[i])
import numpy as np
def funPrint(x):
  print(len(x))
  for i in range(len(x)):
    print(x[i])



class RustModelSim(object):
  def __init__(self,reward,p,beta=0.9999):
    self.S = int(91)
    self.p = np.array(p)
    self.reward = reward
    self.beta = beta
    # A (SxS) matrix indicating the probability of a bus transitioning
    # from a state s to a state s' (used to compute maintenance utility)
    self.trans_mat = np.zeros((self.S, self.S))
    for i in range(self.S):
      for j, _p in enumerate(self.p):
        if i + j < self.S-1:
          self.trans_mat[i][i+j] = _p
        elif i + j == self.S-1:
          self.trans_mat[i][self.S-1] = self.p[j:].sum()
        else:
          pass

  def myopic_costs(self): # - reward function
    rc = self.reward[0]          #F : action 1
    thetas = self.reward[1]     #c : action 0
    maint_cost = np.reshape([-s * thetas for s in range(self.S)],(1,-1))
    repl_cost = np.reshape([-rc for s in range(self.S)],(1,-1))  #action 1
    #print(rc,thetas)
    #print(maint_cost.shape,repl_cost.shape)

    return np.vstack((maint_cost, repl_cost)).T
  
  def fl_costs(self, threshold=1e-4, suppr_output=False): #compute V^n
    
    achieved = True
    
    # Initialization of the contraction mapping
    k = 0
    EV = np.ones((self.S, 1))        
    
    EV_myopic = self.myopic_costs()
    EV_new = np.zeros((self.S, 1))
    
    # Contraction mapping Loop
    while abs(EV_new-EV).max() > threshold:
      EV = EV_new 
      #pchoice = self.choice_prob(EV) #\pi_theta(s,a)
      Q0 = EV_myopic[:,0] + self.beta * self.trans_mat.dot(EV).reshape(-1)
      Q1 = EV_myopic[:,1] + self.beta * EV[0]
      Q = np.vstack((Q0,Q1)).T
      
      min_cost = Q.max(1).reshape(-1,1)
      cost = Q - min_cost
      util = np.exp(cost)
      EV_new =  min_cost + np.log(util.sum(1).reshape(-1,1))
      
      k += 1
      if k == 1000:
        achieved = False
        break
    
    # Output:
    if not suppr_output:
      if achieved:
        print("Convergence achieved in {} iterations".format(k))
      else:
        print("CM could not converge! Mean difference = {:.6f}".format(
            (EV_new-EV).mean()))
    return EV_new,Q


max_iteration = 90
sample_size = 3000
Rc = 20
r = 0.4
theta = np.array([0.3, 0.4, 0.3])
beta=0.9999
rm = RustModelSim([Rc,r],theta,beta = beta)
EV,_ = rm.fl_costs()

print(EV.shape)
print(EV)

states = np.zeros([max_iteration+1,sample_size],dtype=int)
states_tep = np.zeros([max_iteration+1,sample_size],dtype=int)
action = np.zeros([max_iteration,sample_size],dtype=int)

for ss in range(sample_size):
  aId = 0
  for mi in range(max_iteration):
    ds = np.random.choice([0,1,2],p=theta)
    states[mi+1][ss] = states[mi][ss] +ds
    states_tep[mi+1][ss] = states[mi][ss] +ds
    #unobs = np.random.gumbel(-np.euler_gamma, 1, size=2)
    if aId==0:
      st = states[mi+1][ss]
      maint_cost = -r*st  + beta*(theta.dot(np.reshape(EV[st:st+3],(3,1)))[0])
      repl_cost = -Rc  + beta*(theta.dot(np.reshape(EV[0:3],(3,1)))[0])  #action 1
      #maint_cost = -r*st +unobs[0] +beta*(theta.dot(np.reshape(EV[st:st+3],(3,1)))[0])
      #repl_cost = -Rc +unobs[1] +beta*(theta.dot(np.reshape(EV[0:3],(3,1)))[0])  #action 1
      #a0 = states[mi+1][ss] * r
      #if a0 < Rc:
      if repl_cost < maint_cost:
        action[mi][ss] = 0
      else:
        action[mi][ss] = 1
        aId = mi +1
    else:
      states_tep[mi+1][ss] = states[mi+1][ss] - states[aId][ss]
      st = states_tep[mi+1][ss]
      maint_cost = -r * st  +beta*(theta.dot(np.reshape(EV[st:st+3],(3,1)))[0])
      repl_cost = -Rc  +beta*(theta.dot(np.reshape(EV[0:3],(3,1)))[0])  #action 1
#      maint_cost = -r * st +unobs[0] +beta*(theta.dot(np.reshape(EV[st:st+3],(3,1)))[0])
#      repl_cost = -Rc +unobs[1] +beta*(theta.dot(np.reshape(EV[0:3],(3,1)))[0])  #action 1
      #a0 = (states[mi+1][ss] - states[aId][ss])* r
      #if a0 < Rc:
      if repl_cost < maint_cost:
        action[mi][ss] = 0
      else:
        action[mi][ss] = 1
        aId = mi +1
#funPrint(states)
#funPrint(action)
states = np.split(states, [500,1500], axis = 1)
states_tep = np.split(states_tep, [500,1500], axis = 1)
action = np.split(action, [500,1500], axis = 1)
funPrint(states_tep[0])
funPrint(action[0])

import numpy as np
import scipy.optimize as opt

class RustModel(object):
  def __init__(self,ac_worker,st_worker,p,npars):
    self.endog = np.reshape(ac_worker,(-1,1))
    self.exog = np.reshape(st_worker,(-1,1))
    self.npars = npars
    self.N = self.endog.shape[0]
    self.S = int(91)
    self.p = np.array(p)
    # A (SxS) matrix indicating the probability of a bus transitioning
    # from a state s to a state s' (used to compute maintenance utility)
    self.trans_mat = np.zeros((self.S, self.S))
    for i in range(self.S):
      for j, _p in enumerate(self.p):
        if i + j < self.S-1:
          self.trans_mat[i][i+j] = _p
        elif i + j == self.S-1:
          self.trans_mat[i][self.S-1] = self.p[j:].sum()
        else:
          pass
    # A second (SxS) matrix which regenerates the bus' state to 0 with
    # certainty (used to compute the replacement utility)
    self.regen_mat = np.vstack((np.ones((1, self.S)),np.zeros((self.S-1, self.S)))).T

  def myopic_costs(self,params): # - reward function
    rc = params[0]          #F : action 1
    thetas = params[1:]     #c : action 0
    maint_cost = np.reshape([-s * thetas for s in range(0, self.S)],(1,-1))
    repl_cost = np.reshape([-rc for s in range(0, self.S)],(1,-1))  #action 1
    #print(np.shape(maint_cost),np.shape(repl_cost))
    #print(np.shape(np.vstack((maint_cost, repl_cost)).T))

    return np.vstack((maint_cost, repl_cost)).T
  
  def fl_costs(self,params, beta=0.9999, threshold=1e-4, suppr_output=False): #compute V^n
    
    achieved = True
    
    # Initialization of the contraction mapping
    k = 0
    EV = np.ones((self.S, 1))        
    
    EV_myopic = self.myopic_costs(params)
    EV_new = np.zeros((self.S, 1))
    
    # Contraction mapping Loop
    while abs(EV_new-EV).max() > threshold:
      EV = EV_new 
      #pchoice = self.choice_prob(EV) #\pi_theta(s,a)
      Q0 = EV_myopic[:,0] + beta * self.trans_mat.dot(EV).reshape(-1)
      Q1 = EV_myopic[:,1] + beta * EV[0]
      Q = np.vstack((Q0,Q1)).T
      
      min_cost = Q.max(1).reshape(-1,1)
      cost = Q - min_cost
      util = np.exp(cost)
      EV_new =  min_cost + np.log(util.sum(1).reshape(-1,1))
      
      k += 1
      if k == 1000:
        achieved = False
        break
    
    # Output:
    if not suppr_output:
      if achieved:
        print("Convergence achieved in {} iterations".format(k))
      else:
        print("CM could not converge! Mean difference = {:.6f}".format(
            (EV_new-EV).mean()))
    return EV_new,Q
  
  def choice_prob(self,cost_array):  #\pi_theta(s,a)
    cost = cost_array - cost_array.max(1).reshape(-1,1)
    util = np.exp(cost)
    pchoice = util/(np.sum(util, 1).reshape(-1,1))
    return pchoice
        
  def loglike(self,params):
    utilV,utilQ = self.fl_costs(params, suppr_output=True) 
    pchoice = self.choice_prob(utilQ)    
    logprob = 0
    for sample_data in range(self.N):
      ac_tep = int(self.endog[sample_data])
      st_tep = int(self.exog[sample_data])
      logprob += np.log(pchoice[st_tep,ac_tep]) 
    return -logprob   
##################################Rust################################
  def fit_likelihood(self,x0=None,bounds=None):
    if bounds == None:
      bounds = [(1e-6, None) for i in range(self.npars)]
    if x0 == None:
      x0 = [0.1 for i in range(self.npars)]
    fitted = opt.fmin_l_bfgs_b(self.loglike, 
                               x0=x0, 
                               approx_grad=True, 
                                bounds=bounds,
                               maxiter=1)
    return fitted[0]
    
#rm = RustModel(action[0],states_tep[0],theta,2)
#rm.fit_likelihood(x0=[0.1,0.2])
    
import ray
import numpy as np
@ray.remote
class ParameterServer:
    def __init__(self, num_workers, weights_ids):
        self.num_workers = num_workers
        self.weights_ids = weights_ids

    def get_weights_ids(self):
      return self.weights_ids

    def get_num_workers(self):
      return self.num_workers

    def set_weights_ids(self, worker_index, id):
      self.weights_ids[worker_index] = id[0]

@ray.remote

def worker_task(current_worker_index,ps,ac=action,st = states_tep,p=theta,npars=2):
  rm = RustModel(ac[current_worker_index],st[current_worker_index],p,npars)
  
  #return rm.fit_likelihood()


  def get_flocking_potential():
    all_weights_ids = ray.get(ps.get_weights_ids.remote())
    nw = ray.get(ps.get_num_workers.remote())
    #print(all_weights_ids,nw)
    flocking_dis = []
    for fw in range(nw):
      w = ray.get(all_weights_ids[fw])
      flocking_dis.append(w)
    #print(flocking_dis)
    return np.mean(np.array(flocking_dis), axis=0)
  #return get_flocking_potential()[0]    
  step = 0  
  while step < 100:
    f_p = get_flocking_potential()
    #print('x0',f_p)
    new_weights = rm.fit_likelihood(x0=[f_p[0][0],f_p[0][1]])
    #print('x_new',new_weights)
    weights = [[new_weights[0],new_weights[1]]]
    #print('x_new',weights)
    #weights_id = ray.put(new_weights)
    weights_id = ray.put(weights)
    ps.set_weights_ids.remote(current_worker_index, [weights_id])
    
    step += 1
    
    if step % 10 == 0 and current_worker_index == 0:
      print('step', step, 'weights',ray.get(ray.get(ps.get_weights_ids.remote())[0]),ray.get(ray.get(ps.get_weights_ids.remote())[1]))
      #print('step', step, 'weights', new_weights)
  #return ray.get(ray.get(ps.get_weights_ids.remote())[0]),ray.get(ray.get(ps.get_weights_ids.remote())[1])

if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_gpus = 3)
#    print(ray.get(worker_tasks))
##    ray.wait(worker_tasks, num_returns=3)

    init_weight = 0.1*np.ones((1,2))
    #print('\n\n shape is', init_weight.shape, '\n\n')
    weights = [init_weight for _ in range(3)]
    #print(weights)
    weights_ids = [ray.put(w) for w in weights]
    #print(weights_ids)

    ps = ParameterServer.remote(num_workers=3, weights_ids=weights_ids)
    #print(ray.get(ps.get_num_workers.remote()))
    #print(ps)
    worker_tasks = [worker_task.remote(i, ps) for i in range(3)]
    #print(ray.get(worker_tasks[0]))
    ray.wait(worker_tasks, num_returns=3)