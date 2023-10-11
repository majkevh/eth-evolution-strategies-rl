


from em import EM
import argparse
import sys

parser = argparse.ArgumentParser(description= 'Run ES Symulation')
parser.add_argument('-alg', '--alg', type= str, metavar='', required = True, 
                    help= 'Choose Algorithm')
parser.add_argument('-obj', '--objective', type= str, metavar='', required = True, 
                    help= 'Choose Benchmark function bewteen: Rosenbrock, Ackley, Griewank, Hypersphere, Rastrign or RL problem bewteen: "Swimmer-v3", "InvertedPendulum-v2", "Reacher-v2", "Hopper-v3"')
parser.add_argument('-dim', '--dimension', type= int, metavar='', required = False, 
                    help= 'Choose dimensionality')
parser.add_argument('-gen', '--generations', type= int,metavar='', required = True, 
                    help= 'Choose number of generations')
parser.add_argument('-pop', '--population', type= int, metavar='',required =False,
                    help= 'Choose number of individuals in population')
parser.add_argument('-g', '--gamma', type= float, metavar='',required =True, 
                    help= 'Choose smoothing parameter gamma')
parser.add_argument('-nruns', '--numberruns', type= int, metavar='',required =True, 
                    help= 'Select number of runs of algorithm')
parser.add_argument('-method', '--hessianmethod', type= str, metavar='',required =True,
                    help= 'Chose correction method for hessian between: linesearch, evinvert, evfrob, skip,cholcorrection, evdelta, eveucl')
parser.add_argument('-cov', '--covariance', type= str, metavar='',required =False, 
                    help= '"full" if full covariance, "diag" for diagonal covariance')
parser.add_argument('-sim', '--simulation', type= str, metavar='',required =True,
                    help= 'Set to "RL" for reinforcement learning simulation, set to "BM" for Benchmark simulation')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
args = parser.parse_args()

valid_sim= ["RL", "BM"]

if __name__ == '__main__':
    assert args.simulation in valid_sim, "Set to [RL] for reinforcement learning simulation, set to [BM] for Benchmark simulation"
    if (args.simulation == 'RL'):
        SYM = EM(algorithm = args.alg, objective= args.objective, 
                 dim = None, population = None,
                 generations = args.generations, gamma = args.gamma, 
                 number_runs = args.numberruns, method =  args.hessianmethod, 
                 type_covariance = None)  
        SYM.run()
    elif (args.simulation == 'BM'):
        SYM = EM(algorithm = args.alg, objective= args.objective, 
                 dim = args.dimension, population = args.population,
                 generations = args.generations, gamma = args.gamma, 
                 number_runs = args.numberruns, method =  args.hessianmethod, 
                 type_covariance = args.covariance)  
        SYM.run()


 

        
        
        
        
        