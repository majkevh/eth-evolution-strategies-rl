from functools import wraps   

valid_cov = ["full", "diag", None] 
valid_functions = ["Rosenbrock", "Ackley", "Griewank", 
                        "Hypersphere", "Rastrigin"]

valid_method = ["linesearch", "evinvert", "evfrob", "skip", 
                "cholcorrection", "evdelta", "eveucl", "none"]

valid_problems = ["Swimmer-v3", "InvertedPendulum-v2", "Reacher-v2", "Hopper-v3",
                  "Walker2d-v3", "HalfCheetah-v3"]

valid_alg = ["SGA", "EM", "SNM", "EMH", "MAP", "NGA"]
gamma_alg = ["SNM", "EMH", "MAP"]


def check_parameters(func):
    """ Check if parameters of class are valid """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        if kwargs["algorithm"] not in valid_alg:
            raise ValueError(str(kwargs["algorithm"]) + " is not a valid algorithm")
            
        if not ((kwargs["objective"] in valid_functions) or (kwargs["objective"] in valid_problems)):
            raise ValueError(str(kwargs["objective"]) + " is not a valid objective") 
            
        if kwargs["generations"] < 2:
            raise ValueError(str(kwargs["generations"]) + " is not a valid number of maximum generations")
            
        if ((kwargs["gamma"] < 0) or (kwargs["gamma"] > 1)):
            raise ValueError(str(kwargs["gamma"]) + " is not a valid smoothing parameter")
        
        if (kwargs["gamma"] !=1 and (kwargs["algorithm"] not in gamma_alg)):
            print("Attention, only MAP, SNM and EMH can use a momentum parameter, smoothing parameter ignored")
            
        if (kwargs["number_runs"] < 1):
            raise ValueError(str(kwargs["number_runs"]) + " is not a valid number of runs")
            
        if kwargs["method"] not in valid_method:
            raise ValueError(str(kwargs["method"]) + " is not a valid Hessian correction method")
                
        if (kwargs["method"] !="none" and (kwargs["algorithm"] != "SNM")):
            print("Attention, only SNM needs a hessian correction method, parameter ignored")
        
        if kwargs["type_covariance"] not in valid_cov:
            raise ValueError(str(kwargs["type_covariance"]) + " is not a valid covariance type")
            
        if ((kwargs["dim"] is not None) and kwargs["dim"] < 2):
            raise ValueError(str(kwargs["dim"]) + " is not a valid dimension")
        
        if (kwargs["dim"] is None):
            assert kwargs["population"] is None, " Error in population or dimension values"
        
        if (kwargs["population"] is None):
            assert kwargs["dim"] is None, " Error in population or dimension values"
        
        return func(*args, **kwargs)
    return wrapper
