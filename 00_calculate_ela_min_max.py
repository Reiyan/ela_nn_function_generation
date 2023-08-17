import cocoex
import itertools
import numpy as np
import os
import pandas as pd
import pflacco.classical_ela_features as pf
import pflacco.misc_features as mf
import random
import sys

from multiprocessing import Pool
from pflacco.sampling import create_initial_sample

#ROOT = '/scratch/tmp/r_prag01/ela_function_generation'
ROOT = './'

def run_experiment(experiment):
    results = []

    # Initialize BBOB Suite
    suite = cocoex.Suite("bbob", f"instances:{experiment[2]}", f"function_indices:{experiment[0]} dimensions:{experiment[1]}")

    for problem in suite:
        # Get meta information about the opt problem
        fid = problem.id_function
        iid = problem.id_instance
        dim = problem.dimension
        
        for rep in range(100):
            # Set seeds, this ensures, that at least for every (fid,dim,iid) the seeds are different over all x repetitions
            seed = int(fid) * int(iid) * int(dim) * int(rep + 1)
            np.random.seed(seed)
            random.seed(seed)
            
            # Calculate ELA features
            X = create_initial_sample(dim, sample_coefficient = 250, sample_type = 'lhs', lower_bound = -5, upper_bound = 5)
            y = X.apply(lambda x: problem(x), axis = 1)
            y = (y - y.min())/(y.max() - y.min())

            ela_distr = pf.calculate_ela_distribution(X, y)
            ela_meta = pf.calculate_ela_meta(X, y)
            fd = mf.calculate_fitness_distance_correlation(X, y)
            nbc = pf.calculate_nbc(X, y)


            data = pd.DataFrame({**ela_distr, **ela_meta, **fd, **nbc}, index = [0])
            data[['fid', 'dim', 'iid', 'rep']] = [fid, dim, iid, rep]
           
            results.append(data)

    # Consolidate list of potentially multiple dataframes to a single dataframe
    df = pd.concat(results).reset_index(drop = True)
   
    return df


# Function wrapper needed for multiprocessing
if __name__ ==  '__main__':
    if len(sys.argv) == 1:

        fids = range(1, 25)
        dims = [2, 3]
        iids = range(1, 2)

        cart_prod = itertools.product(*[fids, dims, iids])

        # Debug code:
        #for experiment in cart_prod:
        #    run_experiment(experiment)
        
        with Pool(6) as p:
            results = p.map(run_experiment, cart_prod)

        # Consolidate the results of each parallelized process into a single data frame
        data = pd.concat(results).reset_index(drop = True)
        data.to_csv(os.path.join(ROOT, f'02_ela_raw.csv'), index = False)

        cols = [x for x in data.columns if 'costs_runtime' not in x][:-4]
        data_min = data.groupby('dim')[cols].min()
        data_min['type'] = 'min'
        data_max = data.groupby('dim')[cols].max()
        data_max['type'] = 'max'
        pd.concat([data_min, data_max]).to_csv(os.path.join(ROOT, f'02_ela_min_max.csv'))
        
    else:
        raise SyntaxError("Insufficient number of arguments passed")
