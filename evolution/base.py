import numpy as np
from batchnorm import NormalizedEnv
from numpy import linalg as LA
import pandas as pd
from numpy.linalg import inv
import benchmark_functions as bf 
import tensorflow_probability as tfp
import gym
from termcolor import colored
import math 
from scipy.stats import rankdata
from check_constructor import check_parameters

class ES: 

    @check_parameters
    def __init__(self, **kwargs):
        prop_defaults = {
            "algorithm": "NGA", 
            "objective": "Hypersphere",
            "dim": None,
            "generations": 1000,
            "population": None,
            "gamma": 1, 
            "number_runs": 5, 
            "method": "evdelta", 
            "type_covariance": "full",
            "rl": False
        }
        
        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)
        
        if self.dim is None:
            self.rl = True
            _, self.dim = self.chose_objective_function()
            self.initial_mean = np.zeros(self.dim) 
            self.population, self.tolerance, self.type_covariance = self.Mujoco()
            self.hypersigma = 1e-3
        else:
            self.tolerance= 1e-5
            self.target = self.set_target() 
            self.constant_norm = math.sqrt(100/self.dim)+ self.target[0]
            self.initial_mean = self.constant_norm*np.ones(self.dim) 
            self.value_target = self.function(self.target)
            self.hypersigma = 1 
            
        self.N = int(self.dim*(self.dim+1)/2) 
        self.cutoff_worse_evaluations =  int(0.25*self.population) 
        
        if self.type_covariance == "full":
            self.min_population = self.N + self.dim + self.cutoff_worse_evaluations 
            self.duplication = self.D()
            self.identity = np.identity(self.dim)
            self.elimination = self.L()
            self.Natty_Inv= inv(self.elimination.dot(self.Nu()).dot(self.elimination.T))
        else:
            self.min_population = 2*self.dim + self.cutoff_worse_evaluations
            
        assert self.population > self.min_population, f"For the dimension chosen, the population must be at least {self.min_population}"
        self.initial_cov = (self.hypersigma)*np.identity(self.dim) 
        
        self.max_try = 4
            
        self.tol_min_eig = 1e-50
        self.beta1 = 0.8 #ADAM parameter
        self.beta2 = 0.999 #ADAM parameter
        
        if self.algorithm == "NGA":
            self.alpha = .08 
        elif self.algorithm == "SGA":
            self.alpha = .01
            
        self.mean_method_ga = "aravind" #none, adam or aravind     
    
    def reinforce(self, theta, env):
        """Reinforcement learning environment"""
        
    
        actionDim = env.action_space.shape[0]
        stateDim = env.observation_space.shape[0]
        low = np.full(env.action_space.shape, -float("inf"), dtype=np.float64)
        high = np.full(env.action_space.shape, float("inf"), dtype=np.float64)
        step = 0
        done = False
        cumulativeReward = 0.0
        
        assert len(theta) == stateDim * actionDim
        
        thetaMatrix = theta.reshape((actionDim, stateDim))
        state = env.reset().tolist() 


        while not done and step < 1000:
            mapreinforce = np.matmul(thetaMatrix, state)
            action = np.clip(mapreinforce, a_min=low, a_max= high)
            state, reward, done, _ = env.step(action)

            cumulativeReward = cumulativeReward + reward
            step = step + 1

        self.total_time_steps[self.current_gen] += step
        return cumulativeReward
    

    def chose_objective_function(self):
        """Function that return the chosen objective function or the RL environment"""
        
        if not self.rl:
            if(self.objective == bf.Rosenbrock().name()):
                chosen_function = bf.Rosenbrock(n_dimensions=self.dim)
            elif(self.objective == bf.Ackley().name()):
                chosen_function = bf.Ackley(n_dimensions=self.dim)
            elif(self.objective == bf.Griewank().name()):
                chosen_function = bf.Griewank(n_dimensions=self.dim)
            elif(self.objective == bf.Hypersphere().name()):
                chosen_function = bf.Hypersphere(n_dimensions=self.dim)
            elif(self.objective == bf.Rastrigin().name()):
                chosen_function = bf.Rastrigin(n_dimensions=self.dim)
            return chosen_function
        else:
            envName = self.objective
            env = gym.make(envName)
            env = NormalizedEnv(env, ob=True, ret=False)
            
            actionDim = env.action_space.shape[0]
            self.stateDim = env.observation_space.shape[0]
            dim = actionDim*self.stateDim
            obj_func = lambda x: self.reinforce(x, env)
            
            return obj_func, dim
    

    def set_target(self):
        """Returns the optimum point for benchmark optimization functions"""
        
        return np.array(self.chose_objective_function().minimum())


    def Mujoco(self):
        """Returns tolerance and population (prechosen) for RL tasks"""
        
        if self.objective == "Swimmer-v3":
            return 250, 350, "full"
        elif self.objective == "InvertedPendulum-v2":
            return 30, 1000, "full"
        elif self.objective == "Reacher-v2":
            return 280, -2, "full"
        elif self.objective == "Hopper-v3":
            return 970, 3400, "full"
        elif self.objective == "HalfCheetah-v3":
            return 350 , 2500, "diag"
        elif self.objective == "Walker2d-v3":
            return 350, 4000, "diag"
        
    
    
    def function(self, point): 
        """Return function evaluations or cumulative reward (RL)"""
        
        if not self.rl:
            point = point.astype('float64')
            func = self.chose_objective_function()
            return -func(point)
        else:
            func, _ = self.chose_objective_function()
            return func(point)


    def Initialize(self):
        """Initializes the parameters at the beginning of each simulation"""
        
        self.cov = self.initial_cov
        self.mean = self.initial_mean
        self.linalgerror = False
        self.total_time_steps = np.zeros(self.generations)
        self.best_point = np.zeros(self.generations)
        self.first_moment_mean = np.zeros(self.dim)
        self.second_moment_mean = np.zeros(self.dim)
        self.error = np.zeros((self.generations, 1)) 
        self.max_eig = np.zeros((self.generations, 1))
        self.min_eig = np.zeros((self.generations, 1)) 
        self.cond_number = np.zeros((self.generations, 1)) 
        self.mean_container = np.zeros((self.generations,self.dim)) 
    
    def vechtorize(self, matrix):
        """Vechtoriaztion of a matrix"""
        
        return matrix[np.tril_indices(self.dim)]
    
    
    def mathricize(self, vector):
        """Returns the mathtrization of a vector (inverse opeation of vech())"""
        
        temp = (self.duplication).dot(vector)
        return np.tril(np.reshape(np.ravel(temp), (self.dim, self.dim)).T)
    
    
    def Expectation(self, normalize):
        """Expectation step in the EM algorithm: 
            returns the normalized and ranked function evaluations"""

        function_evaluation = np.apply_along_axis(self.function, axis=0, arr=self.z)
        
        if self.rl:
            self.best_point[self.current_gen] = np.median(function_evaluation)
        else:
            self.best_point[self.current_gen] = np.max(function_evaluation)
        function_evaluation = rankdata(function_evaluation)
        function_evaluation = np.where(function_evaluation<self.cutoff_worse_evaluations, 0, function_evaluation)
        if normalize:
            sumnorm = np.sum(function_evaluation)
            function_evaluation = function_evaluation/sumnorm
        return function_evaluation
    
    
    def StoppingError(self):
        """Return True if matrix is no more positive semi-definite or
            no improvement in obj function for 100 generations or divergence"""
            
        if ((not self.rl) and (self.current_gen >100)):
            isdiverged = (((np.linalg.norm(self.mean-self.target))>=(np.linalg.norm(self.mean_container[0, :]-self.target))))
            if (isdiverged and ((self.algorithm!= "EM") or (self.algorithm != "SGA"))):
                print(colored("WARNING: evaluation stopped due to divergence", "yellow"))
                return True
            elif (((np.linalg.norm(self.mean-self.target))>=(np.linalg.norm(self.mean_container[self.current_gen-100, :]-self.target))) and ((self.algorithm== "EM") or (self.algorithm == "SGA"))):
                print(colored("WARNING: evaluation stopped due to no improvement in obj function for 100 generations", "yellow"))
                return True
            
        if (self.min_eig[self.current_gen] <= self.tol_min_eig):
            print("\n")
            print(colored("WARNING, matrix is no more positive definite", "red"))
            print(colored("Last generation: " +str(self.current_gen), "red"))
            print(colored("Mean at generation " +str(self.current_gen)+ " is " + str(self.mean), "red"))
            print(colored("Minimum eigevalue at this generations is " +str(self.min_eig[self.current_gen]), "red"))
            print(colored("Covariance at generation " +str(self.current_gen)+ " is " , "red"))
            print(colored(pd.DataFrame(self.cov), "red"))
            return True
    
    
    def StoppingCriterium(self):
        """Return True if stopping criterium reached"""
        
        if not self.rl:
            if((self.current_gen == self.generations-1) or (np.linalg.norm(self.mean-self.target)< self.tolerance)):
                if self.current_gen != self.generations -1:
                    print("\n")
                    print(f"Desired tolerance already reached, algorithm early stopped at generation: {self.current_gen}")
                else:
                    print("\n")
                    print("Reached maximum generation")
                return True
        else:
            if(self.best_point[self.current_gen] >= self.tolerance):
                return True
        

    
    def ExactUpdate(self, weights):
        """Update of the covariance and mean for the MAP/MLE/EMH algorithms"""
        
        if self.algorithm == "EMH":
            self.CovUpdate(weights)
            self.MeanUpdate(weights)
            
        elif self.algorithm == "EM":
            self.MeanUpdate(weights)
            self.CovUpdate(weights)
            
        elif self.algorithm == "MAP":
            old_mean = self.mean.copy()
            self.MeanUpdate(weights)
            z_dm =(self.z.T-old_mean).T
            if self.type_covariance == "full":  
                self.cov = (1.0 -self.gamma)*np.array(self.cov) + self.gamma*np.einsum('ij, kj, j->ik',z_dm,z_dm, weights) + (2*self.gamma-1)*np.outer(old_mean, old_mean)- np.outer(self.mean, self.mean)+2*np.outer(self.mean, old_mean)
            else:
                self.cov = (1.0 -self.gamma)*np.array(self.cov) + self.gamma*np.diag(np.einsum('ij, ij, j->i',z_dm,z_dm, weights))+ np.diag((2*self.gamma-1)*(old_mean**2)- self.mean**2 + 2*self.mean*old_mean)
                
    
    def CovUpdate(self, weights):
        """Helper function for self.ExactUpdate(weights)"""
        
        z_dm =(self.z.T-self.mean).T
        if self.type_covariance == "full":
            self.cov = (1.0 -self.gamma)*np.array(self.cov) + self.gamma*np.einsum('ij, kj, j->ik',z_dm,z_dm, weights)
        else:
            self.cov =(1.0 -self.gamma)*np.array(self.cov) + self.gamma*np.diag(np.einsum('ij, ij, j->i',z_dm,z_dm, weights))
            
    
    def MeanUpdate(self, weights):
        """Updates the mean for the MAP/MLE/EMH/SNM algorithm"""
        
        self.mean = (1.0-self.gamma)*np.array(self.mean) + self.gamma*np.einsum('i,ji->j', weights, self.z)  

        
    def Inverse_correction(self, inverse, EV, EF):
         """Returns the corrected inverse of the hessian in the SNM algorithm"""
        
         if self.method == "linesearch":
             inverse = inverse + abs(np.min(EV))*np.identity(self.N)
         elif self.method == "cholcorrection":
             inverse = np.array(tfp.experimental.linalg.simple_robustified_cholesky(
                 inverse, tol=1e-6, name='simple_robustified_cholesky'))
             inverse = inverse.dot(inverse.T)
         elif self.method == "evinvert":
             EV[EV<0] = -EV[EV<0]
             inverse = EF.dot(np.diag(EV)).dot(EF.T)
         elif self.method == "evdelta":
             EV[EV<0] = self.tol_min_eig
             inverse = EF.dot(np.diag(EV)).dot(EF.T)
         elif self.method == "evfrob":
             if(self.type_covariance == "full"):
                 tau = np.zeros(self.N)
             else:
                 tau = np.zeros(self.dim)
             for i in range(len(EV)):
                 if EV[i]<0:
                     tau[i] = self.tol_min_eig - EV[i]
             inverse = EF.dot(np.diag(EV)+np.diag(tau)).dot(EF.T)
         elif self.method == "eveucl":
             tau = max(0, self.tol_min_eig-np.min(EV))
             if(self.type_covariance == "full"):
                 identity = np.identity(self.N)
             else:
                 identity = np.identity(self.dim)
             inverse = inverse + tau*identity
         return inverse
 
    
    def SNMCovUpdate(self, weights):
        """Computes the update of the covariance matrix in the SNM algorithm"""
        
        if self.type_covariance == "full":
            try:
                self.S = np.linalg.cholesky(self.cov)
            except np.linalg.LinAlgError as e:
                if "Matrix is not positive definite" in str(e):
                    self.linalgerror= True
                    return
            s = self.vechtorize(self.S)
            dJ_dSIGMA = self.dJ_dSIGMA(weights) 
            evaluation = dJ_dSIGMA.dot(self.duplication)
            R = self.R(evaluation)
            Q = self.Q()
            FIM = self.FIM()
            
            inverse = inv(R+ Q.dot(FIM).dot(Q.T))
            EV, EF = LA.eig(inverse)
    
            if np.any(EV<=0) and self.method != "skip":
                inverse = self.Inverse_correction(inverse, EV, EF)
            elif np.any(EV<=0) and self.method == "skip":
                return 
            s = s - inverse.dot(Q).dot(evaluation)
            self.S = self.mathricize(s)
            
            self.cov = (1-self.gamma)*self.cov + self.gamma*self.S.dot(self.S.T)
        
        else:
            z_dm= (self.z.T-self.mean).T
            s = np.diag(np.sqrt(self.cov))
            dersum = np.einsum('ij, ij, j->i',z_dm,z_dm, weights)
            s = s- (1/s - 1/(s**3)*dersum)/(3/(s**2)-1/(s**4)*dersum)
            self.cov = (1-self.gamma)*self.cov + self.gamma*(np.diag(s)**2)
            
    

    def GACovUpdate(self, weights):
        """Computes the update of the covariance matrix in the SGA/NGA algorithm"""
        
        if self.type_covariance == "full":
            try:
                self.S = np.linalg.cholesky(self.cov)
            except np.linalg.LinAlgError as e:
                if "Matrix is not positive definite" in str(e):
                    self.linalgerror= True
                    return 
            s = self.vechtorize(self.S)
            derivative = self.dJ_ds(weights)
            
            if self.algorithm == "NGA":
                FIM_INV = self.Natty_Inv_FIM()
                search_direction = FIM_INV.dot(derivative)
            else:
                search_direction = derivative
        else:
            s = np.sqrt(np.diag(self.cov))
            z_dm= (self.z.T-self.mean).T
            if self.algorithm =="NGA":
                search_direction = -s/2 + 1/(2*s)*np.einsum('ij, ij, j->i',z_dm,z_dm, weights)
            else:
                search_direction = -1/s + 1/(s**3)*np.einsum('ij, ij, j->i',z_dm,z_dm, weights)

        s = s + self.alpha*search_direction
        
        if self.type_covariance == "full":
            self.S = self.mathricize(s)
        else:
            self.S = np.diag(s)
        
        self.cov = self.S.dot(self.S.T)
    

    def GAMeanUpdate(self, weights):
        """Computes the update of the mean in the SGA/NGA algorithm, 
            one could chose between classic and ADAM update method"""
        
        if self.algorithm == "NGA" and self.mean_method_ga != "aravind":
            search_direction =np.einsum('i,ji->j', weights, self.z)-self.mean
        else:
            z_dm =(self.z.T-self.mean).T
            invcov = inv(self.cov)
            search_direction = np.einsum('i,jk,ki->j', weights, invcov, z_dm)
            
        if self.mean_method_ga == "adam":
            self.first_moment_mean = self.beta1*self.first_moment_mean + (1-self.beta1)*search_direction
            self.second_moment_mean = self.beta2*self.second_moment_mean + (1-self.beta2)*search_direction**2
            first_hat =self.first_moment_mean/(1-self.beta2**(self.current_gen+1))
            second_hat = self.second_moment_mean/(1-self.beta2**(self.current_gen+1))
            
            self.mean = self.mean +self.alpha*first_hat/(np.sqrt(second_hat)+1e-8)
            
        elif self.mean_method_ga == "aravind":
            if self.algorithm == "NGA":
                g = self.cov.dot(search_direction)
                learning_rate = math.sqrt(self.alpha/((search_direction.T).dot(g)))
            else:
                g = search_direction
                learning_rate = (self.alpha/np.linalg.norm(g))
                
            self.mean = self.mean+ learning_rate*g
            
        else:
            self.mean = self.mean+ self.alpha*search_direction


    
    def D(self):
      """Returns the duplication matrix given the dimensionality of the optimization problem"""
        
      vec = np.indices((self.dim, self.dim)).transpose(0,2,1).reshape(2, -1)
      vech = np.zeros((2,self.N))
      vech[0,:], vech[1,:] = np.tril_indices(self.dim)
      chain = np.zeros((self.dim**2, self.N))
      for row in range(self.dim**2):
          for column in range(self.N):
              if(((vec[0, row] == vech[0, column]) and (vec[1, row]== vech[1, column]))or ((vec[1, row] == vech[0, column]) and (vec[0, row]== vech[1, column]))):
                  chain[row, column] = 1
      return chain 

    def L(self):
        """Returns the elimination matrix given the dimensionality of the optimization problem"""        
        
        vec = np.indices((self.dim, self.dim)).transpose(0,2,1).reshape(2, -1)
        vech = np.zeros((2,self.N))
        vech[0,:], vech[1,:] = np.tril_indices(self.dim)
        chain = np.zeros((self.dim**2, self.N))
        for row in range(self.dim**2):
            for column in range(self.N):
                if(((vec[0, row] == vech[0, column]) and (vec[1, row]== vech[1, column]))):
                    chain[row, column] = 1
        return chain.T  

    def Nu(self):
        """Returns the modified-commutation matrix given the dimensionality of the optimization problem"""
        
        A = np.reshape(np.arange(self.dim**2),(self.dim,self.dim),order = 'F')
        w = np.reshape(A.T,self.dim**2, order = 'F')
        M = np.eye(self.dim**2)
        M = M[w,:]
        return (M+ np.identity(self.dim**2))/2


    def Q(self):
        """Returns the matrix Q that appears in the chain rule computation 
            of the derivative of the Gaussian distribution w.r.t the vechtorized 
            Cholesky decomposition of the covariance matrix"""
        
        Q = np.zeros((self.N, self.N))
        vech = np.zeros((2,self.N))
        vech[0,:], vech[1,:] = np.tril_indices(self.dim)
        for row in range(self.N):
            for column in range(self.N):
                i, j = int(vech[0,column]), int(vech[1,column]) #sigma_j
                l, m = int(vech[0,row]), int(vech[1,row]) #s_i
                term = 0  
                if((l == j) and (j != i)):
                    term = self.S[i, m]
                elif((l == j) and (j == i)):
                    term = 2* self.S[i, m]
                elif((l == i) and (i != j)):
                    term = self.S[j, m]
                Q[row, column] = term
        return Q
        
    
    def dJ_dSIGMA(self, weights):
        """Returns the derivative of the Gaussian distribution w.r.t 
            the vectorized covariance matrix"""
        
        invcov = inv(self.cov)
        z_dm = (self.z.T-self.mean).T
        emp_cov = np.einsum('ij, kj, j->ik',z_dm,z_dm, weights)
        return np.reshape(-1/2*invcov.dot(emp_cov-self.cov).dot(invcov), -1)
    
    def dJ_ds(self, weights):
        """Returns the derivative of the Gaussian distribution w.r.t the vechtorized 
            Cholesky decomposition of the covariance matrix"""
            
        inv_s = inv(self.S)
        z_dm = (self.z.T-self.mean).T
        dersum = np.einsum('ij, kj, j, li->kl',z_dm,z_dm, weights, inv_s)
        return self.vechtorize((inv_s.T).dot(inv_s.dot(dersum)-np.identity(self.dim)))
    

    def R(self, evaluation):
        """Returns the matrix R that appears in the chain rule computation 
            of the hessian of the Gaussian distribution w.r.t the vechtorized 
            Cholesky decomposition of the covariance matrix"""
            
        R = np.zeros((self.N, self.N))
        vech = np.zeros((2,self.N))
        vech[0,:], vech[1,:] = np.tril_indices(self.dim)
        for row in range(self.N):
            for column in range(self.N):
                q, r = int(vech[0,column]), int(vech[1,column]) #s_j
                l, m = int(vech[0,row]), int(vech[1,row]) #s_i
                term = 0
                for p in range(self.N):
                    temp = 0
                    if(r == m):
                        i, j = int(vech[0,p]), int(vech[1,p]) #sigma_p
                        if((l == j) and (j != q) and (q==i)):
                            temp = 1
                        elif((l == q) and (q == i) and (i == j)):
                            temp = 2
                        elif((l == i) and (i != q) and (q==j)):
                            temp = 1
                    term = term + temp*evaluation[p]
                R[row, column] = term
        return R

    
    def FIM(self):
        """Returns the Fisher Information matrix of the Gaussian distribution 
            w.r.t the vechtorized covariance matrix"""
            
        invcov = inv(self.cov)
        FIM_4 =  1/2*(np.kron(invcov,invcov))
        return (self.duplication.T).dot(FIM_4).dot(self.duplication)
    
    def Natty_Inv_FIM(self):
        """Returns the inverse of theFisher Information matrix of the Gaussian distribution 
            w.r.t the vechtorized Cholesky decomposition of the covariance matrix"""
            
        return 1/2*self.elimination.dot(np.kron(self.identity, self.S)).dot(self.elimination.T).dot(self.Natty_Inv).dot(self.elimination).dot(np.kron(self.identity, self.S.T)).dot(self.elimination.T)
    


    def Fillfeatures(self):
        """Fills are the feature vectors with the last significant entry after 
            a simulation is correctly finished"""
            
        self.best_point[self.current_gen:] = self.best_point[self.current_gen]
        self.error[self.current_gen:] = self.error[self.current_gen]
        self.mean_container[self.current_gen:] = self.mean_container[self.current_gen]
        self.cond_number[self.current_gen:] = self.cond_number[self.current_gen]
        self.max_eig[self.current_gen:] = self.max_eig[self.current_gen]
        self.min_eig[self.current_gen:] = self.min_eig[self.current_gen]
        self.total_time_steps[self.current_gen:] = self.total_time_steps[self.current_gen]
                    
 
    def Features(self):
        """Fills are the feature vectors with the current values of the features"""
            
        EV, EF = LA.eig(self.cov)
        self.max_eig[self.current_gen] = np.max(EV)
        self.min_eig[self.current_gen] = np.min(EV)
        self.cond_number[self.current_gen]= np.linalg.cond(self.cov)
        if not self.rl:
            self.error[self.current_gen] = np.linalg.norm(self.target- self.mean)
            
            
            
            
            
            
            
            
            
            
            
            
            
