from base import ES
import numpy as np
from tqdm import tqdm
import pandas as pd
from termcolor import colored

class EM(ES):
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def ES_EM(self):
        self.Initialize()

        with tqdm(total=self.generations, position=0, leave=True, ascii=' =', 
                  bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}') as pbar:
            for self.current_gen in range(self.generations):
                
                self.z =  (np.random.multivariate_normal(self.mean, self.cov, self.population)).T
                self.mean_container[self.current_gen, :] = self.mean
                weights = self.Expectation(True)
                
                if self.current_gen != 0 and self.rl:
                    self.total_time_steps[self.current_gen] += self.total_time_steps[self.current_gen-1]
                    
                if self.algorithm == "SGA" or self.algorithm == "NGA":
                    self.GACovUpdate(weights)
                    self.GAMeanUpdate(weights)
                    
                elif self.algorithm == "SNM":
                    self.SNMCovUpdate(weights)
                    self.MeanUpdate(weights)
                    
                else:
                    self.ExactUpdate(weights)

                self.cov = (self.cov+self.cov.T)/2
                self.Features()
                pbar.update()
                if ((self.current_gen == self.generations-1) or self.StoppingCriterium()):
                    pbar.close()
                    if self.current_gen == self.generations-1:
                        if self.rl:
                            print("Simulation ended correctly, reached cumulative reward of "+ str(self.best_point[self.current_gen]))
                        else:
                            print("Searched maximum: " +str(self.target))
                            print("Reached maximum: " +str(self.mean))
                    else:
                        print(colored("Simulation stopped correctly due to early convergence", "green"))
                    print("\n")
                    self.Fillfeatures()
                    return False
                
                elif(self.StoppingError() or self.linalgerror):
                    self.Fillfeatures()
                    pbar.close()
                    return True

    def run(self):
        """Function that runs multiple simulations and export the data at the end"""

        error_runs = np.zeros((self.generations, self.number_runs))
        reward_runs = np.zeros((self.generations, self.number_runs))
        time_steps_runs = np.zeros((self.generations, self.number_runs))
        for run in range(self.number_runs):
            reset = True
            current_try = 0
            print(f"Running {self.algorithm} algorithm, in {self.dim} dimensions for {self.objective}, covariance is {self.type_covariance}")
            print("Run number " +str(run+1)+"/"+str(self.number_runs)+":")
            
            while(reset and (current_try <= self.max_try)):
                if current_try >0:
                    print("Re-try " + "#" + str(current_try) + " out of " + str(self.max_try))
                reset = self.ES_EM()
                current_try = current_try + 1
            self.SaveCSV(run)
            if self.number_runs > 1:
                reward_runs[:, run] = self.best_point
                error_runs[:, run] = self.error[:, 0]
                time_steps_runs[:, run] = self.total_time_steps
        if (self.number_runs > 1):
            if not self.rl:
                self.SaveErrorQuantile(error_runs, np.abs(reward_runs))
            else:
                self.SaveRewardQuantile(reward_runs, time_steps_runs)
                
    def SaveCSV(self, run):
        """Export to .csv file of the meaningful parameters of each simulation"""
            
        dfs = pd.DataFrame(self.best_point, columns = ["best_point"])
        dfs["max_eig"] = self.max_eig
        dfs["min_eig"] = self.min_eig
        dfs["cond_number"] = self.cond_number
        
        if not self.rl:
            dfs["error"] = self.error
        for dim in range(self.dim):
            dfs[f"mean_dim{dim+1}"] = self.mean_container[:, dim]

        dfs.to_csv(f"simulation/raw/{self.algorithm}_{self.gamma}_{self.population}_{self.objective}_{self.type_covariance}_{run+1}.csv")
    
    
    def SaveErrorQuantile(self, error_runs, reward_runs):
        """Export to .csv file the 20%, 50%, 80% quantile of the error and the bestpoint 
            for an ensamble of run of simulations"""
        
        quantile_error = np.zeros((self.generations, 6))
        quantile_error[..., 0] = np.quantile(error_runs, 0.2, axis=1)
        quantile_error[..., 1] = np.quantile(error_runs, 0.5, axis=1)
        quantile_error[..., 2] = np.quantile(error_runs, 0.8, axis=1) 
        quantile_error[..., 3] = np.quantile(reward_runs, 0.2, axis=1)
        quantile_error[..., 4] = np.quantile(reward_runs, 0.5, axis=1)
        quantile_error[..., 5] = np.quantile(reward_runs, 0.8, axis=1) 
        df = pd.DataFrame(quantile_error, columns = ['q20e','q50e','q80e', 'q20b','q50b','q80b'])
        df.to_csv(f"simulation/error/{self.algorithm}_{self.gamma}_{self.dim}_{self.population}_{self.objective}_{self.method}_{self.type_covariance}.csv")
    
    
    def SaveRewardQuantile(self, reward_runs, time_steps_runs):
        """Export to .csv file the 20%, 50%, 80% quantile of the cumulative reward
            for an ensamble of run of simulations"""
        
        total_time_steps_each_run = time_steps_runs[self.generations-1, :]
        ind = np.argsort(total_time_steps_each_run)[len(total_time_steps_each_run)//2]
        
        quantile_reward = np.zeros((self.generations, 4))
        quantile_reward[..., 0] = np.quantile(reward_runs, 0.2, axis=1)
        quantile_reward[..., 1] = np.quantile(reward_runs, 0.5, axis=1)
        quantile_reward[..., 2] = np.quantile(reward_runs, 0.8, axis=1) 
        quantile_reward[..., 3] = time_steps_runs[:, ind]
        df = pd.DataFrame(quantile_reward, columns = ['q20','q50','q80', 'time_steps'])
        df.to_csv(f"simulation/reward/{self.algorithm}_{self.gamma}_{self.dim}_{self.population}_{self.objective}_{self.method}_{self.type_covariance}.csv")               
                
                
                
                
                
                
                
                
                
                
        
