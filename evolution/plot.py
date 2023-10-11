import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse
import os
import sys
import seaborn as sns

parser = argparse.ArgumentParser(description= 'Plotting simulations')
parser.add_argument('-gen', '--generations', type= int, metavar='',required =True, help= 'Choose maximum generations in plot')
parser.add_argument('-del', '--delete', type= int, metavar='',required =True, help= 'Delete files after simulation?')
parser.add_argument('-sim', '--simulation', type= str, metavar='',required =True, help= 'Set to "RL" for reinforcement learning simulation, set to "BM" for Benchmark simulation')
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
args = parser.parse_args()


valid_sim= ["RL", "BM"]

def AnalizeFilename(string):
    """Helper function to get the name and additional data for each raw data file:
        algorithm_gamma_dim_population_objfunction_method_typecov"""
    
    first_split= string.split('/')
    string = first_split[2]
    second_split = string.split('.csv')
    string = second_split[0]
    splitted = string.split('_')
    return [str(splitted[0]), float(splitted[1]), int(splitted[2]), int(splitted[3]), 
            str(splitted[4]) , str(splitted[5]), str(splitted[6])]


def DecideColor(alg, gamma, met):
    """Select the color for the plot of each algorithm"""
    palette = sns.color_palette()
    
    if alg == "SNM":
        color = palette[1]
    elif alg == "MAP":
        color = palette[0]
    elif alg == "EMH":
        color = palette[5]
    elif alg =="SGA":
        color = palette[3]
    elif alg == "EM":
        color = palette[4]
    elif alg == "NGA":
        color = palette[2]
      
    if alg == "MAP" or alg == "EMH" :    
        lab = f"{alg}-{gamma}"
        
    elif alg == "SNM":
        lab = f"{alg}-{gamma}"
        
    else:
        lab = f"{alg}"
            
    return color, lab

def PlotError(max_gen, deletefiles):
    """Plots the error of a given algorithm on the total function evaluations"""
    
    path = "simulation/error"
    
    files = glob.glob(path + "/*.csv")
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    ax.set_ylabel('$L^2$ $Error$')
    ax.set_yscale('log')
    gen = np.arange(1, max_gen+1)
    for filename in files:
        df = pd.read_csv(filename)
        columns = list(df.columns)
        alg, gamma, dim, pop, func, met, type_= AnalizeFilename(filename)
        
        color, lab = DecideColor(alg, gamma, met)
        
        ax.fill_between(gen, np.array(df.loc[:, columns[1]]), np.array(df.loc[:, columns[3]]), alpha=0.2, color = color)
        ax.plot(gen, np.array(df.loc[:, columns[2]]), label=lab, c=color)
    
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(f"simulation/plots/{func}_{dim}D_POP{pop}_{type_}_error.png")
    
    if deletefiles:
        files1 = glob.glob('SIM/error/*')
        files2 = glob.glob('SIM/simulationBM/*')
        for f1 in files1:
            os.remove(f1)
        for f2 in files2:
            os.remove(f2)

def PlotBestvalue(max_gen, deletefiles):
    """Plots the best point of a given algorithm on the total function evaluations"""
    
    path = "simulation/error"
    
    files = glob.glob(path + "/*.csv")
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    ax.set_ylabel('Best objective')
    ax.set_yscale('log')
    gen = np.arange(1, max_gen+1)
    for filename in files:
        df = pd.read_csv(filename)
        columns = list(df.columns)
        alg, gamma, dim, pop, func, met, type_= AnalizeFilename(filename)
        
        color, lab = DecideColor(alg, gamma, met)
        
        ax.fill_between(gen, np.array(df.loc[:, columns[4]]), np.array(df.loc[:, columns[6]]), alpha=0.2, color = color)
        ax.plot(gen, np.array(df.loc[:, columns[5]]), label=lab, c=color)
    
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(f"simulation/plots/{func}_{dim}D_POP{pop}_{type_}_best.png")
    
    if deletefiles:
        files1 = glob.glob('SIM/error/*')
        files2 = glob.glob('SIM/simulationBM/*')
        for f1 in files1:
            os.remove(f1)
        for f2 in files2:
            os.remove(f2)
            
def PlotReward(max_gen, deletefiles):
    """Plots the cumulative reward of a given MuJoCo optimization task on the total generations"""
    
    path = "simulation/reward"
    
    files = glob.glob(path + "/*.csv")
    fig, ax = plt.subplots()
    ax.set_xlabel('Cumulative number of time steps')
    ax.set_ylabel('Cumulative Reward')
    for filename in files:
        df = pd.read_csv(filename)
        columns = list(df.columns)
        alg, gamma, dim, pop, func, met, type_= AnalizeFilename(filename)
        
        color, lab = DecideColor(alg, gamma, met)
        
        ax.fill_between(np.array(df.loc[:, columns[4]]), np.array(df.loc[:, columns[1]]), np.array(df.loc[:, columns[3]]), alpha=0.2, color = color)
        ax.plot(np.array(df.loc[:, columns[4]]), np.array(df.loc[:, columns[2]]), label=lab, c=color)
        
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.rc('grid', linestyle="-", color='black')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(f"simulation/plots/{func}_POP{pop}_{type_}.png")
    
    if deletefiles:
        files1 = glob.glob('SIM/reward/*')
        files2 = glob.glob('SIM/simulationRL/*')
        for f1 in files1:
            os.remove(f1)
        for f2 in files2:
            os.remove(f2)


if __name__ == '__main__':
    if args.simulation == "RL":
        PlotReward(args.generations, args.delete)
    else:
        PlotError(args.generations, args.delete)
        PlotBestvalue(args.generations, args.delete)









