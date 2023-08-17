import cma
import cocoex
import itertools
import numpy as np
import os
import pandas as pd
import pflacco.classical_ela_features as pf
import pflacco.misc_features as mf
import random
import sys

from config import BUDGET
from multiprocessing import Pool
from pflacco.sampling import create_initial_sample
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

#ROOT = '/scratch/tmp/r_prag01/ela_function_generation'
#MIN_MAX_PATH = '/home/r/r_prag01/ela_function_generation/02_ela_min_max.csv'

ROOT = './'
MIN_MAX_PATH = './02_ela_min_max.csv'
ELA_VECTOR_PATH = './median_ela_vectors.csv'

min_max = pd.read_csv(MIN_MAX_PATH)

ela_col_names = ['ela_distr.skewness', 'ela_meta.quad_simple.adj_r2', 'fitness_distance.fitness_std', 'ela_meta.lin_w_interact.adj_r2', 'nbc.nn_nb.sd_ratio', 'nbc.nb_fitness.cor',\
            'ela_meta.lin_simple.adj_r2', 'ela_meta.quad_w_interact.adj_r2']
ela_vectors = pd.read_csv(ELA_VECTOR_PATH)

def run_experiment(experiment):
    results = []

    # Initialize BBOB Suite
    suite = cocoex.Suite("bbob", f"instances:{experiment[2]}", f"function_indices:{experiment[0]} dimensions:{experiment[1]}")

    for problem in suite:
        # Get meta information about the opt problem
        fid = problem.id_function
        iid = problem.id_instance
        dim = problem.dimension
        min_vector = min_max[(min_max.type == 'min') & (min_max.dim == dim)][ela_col_names].to_numpy()
        max_vector = min_max[(min_max.type == 'max') & (min_max.dim == dim)][ela_col_names].to_numpy()
        
        # Set seeds, this ensures, that at least for every (fid,dim,iid) the seeds are different over all x repetitions
        seed = int(fid) * int(iid) * int(dim) #*(rep + 1))
        np.random.seed(seed)
        random.seed(seed)
        
        X = create_initial_sample(dim, sample_coefficient = 250, sample_type = 'lhs', lower_bound = -5, upper_bound = 5)
        y = X.apply(lambda x: problem(x), axis = 1)
        y = (y - y.min())/(y.max() - y.min())

        given_ela_features = ela_vectors[(ela_vectors.fid == experiment[0]) & (ela_vectors.dim == experiment[1])][ela_col_names].to_numpy()[0]
        given_ela_features = (given_ela_features - min_vector)/(max_vector - min_vector)

        # Objective function which calculates the distance between ELA features.
        # The first set of ELA features is calculated on the given sample X and y, where the latter is the intermediate result of CMA-ES.
        # The second set of ELA features is provided externally by some means. In this example, we generate them by sampling a BBOB function and extracting the respective ELA features
        def _objective_function(y_hat, X = X, given_ela_features = given_ela_features):
            if y_hat.ndim == 2:
                y_hat = y_hat.flatten()
            pop_skewness = pf.calculate_ela_distribution(X, y_hat)['ela_distr.skewness']
            tmp_meta = pf.calculate_ela_meta(X, y_hat)
            pop_fd_std = mf.calculate_fitness_distance_correlation(X, y_hat)['fitness_distance.fitness_std']
            tmp_nbc = pf.calculate_nbc(X, y_hat)
            pop_ela_features = np.array([pop_skewness, tmp_meta['ela_meta.quad_simple.adj_r2'], pop_fd_std,\
                tmp_meta['ela_meta.lin_w_interact.adj_r2'], tmp_nbc['nbc.nn_nb.sd_ratio'], tmp_nbc['nbc.nb_fitness.cor'], tmp_meta['ela_meta.lin_simple.adj_r2'], tmp_meta['ela_meta.quad_w_interact.adj_r2']])
            pop_ela_features = (pop_ela_features - min_vector)/(max_vector - min_vector)

            dist = np.array([(pop_ela_features[x] - given_ela_features[x]) ** 2 for x in range(len(given_ela_features))]).mean()
            return dist

        # Generate random cloud of points for y, which serves as starting point for CMA-ES
        x0 = np.random.rand(1, 250 * dim)
        
        # On average, the sample in each dimension attains a value of 0.5, to make sure the entire objective space can be reached,
        # the initial value for sigma has to be ensure 0.5 - 3*sigma = 0 and 0.5 + 3*sigma = 1
        sigma = 0.5/3
        es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': [0, 1], 'ftarget': 0, 'maxfevals': BUDGET * dim, 'verb_disp': 1})
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [_objective_function(x) for x in solutions])
            es.disp()

        # Extract x_best and f_vest found by CMA-ES
        y_hat = es.result[0]
        ela_feature_distance = es.result[1]

        # Save sample to disk
        sample = X.copy()
        sample['y'] = y_hat

        os.makedirs(os.path.join(ROOT, 'samples'), exist_ok = True)
        sample.to_csv(os.path.join(ROOT, 'samples', f'f{fid}_d{dim}_i{iid}_cma_sample_3.csv'), index = False)

        data = pd.DataFrame([[fid, dim, iid, ela_feature_distance]], columns= ['fid', 'dim', 'iid', 'obj_value'])
        data[['set_skewness', 'set_quad_simp_r2', 'set_fd', 'set_lin_inter_r2', 'set_nbc_sd', 'set_nbc_cor', 'set_lin_simple_r2', 'set_quad_inter_r2']] = given_ela_features

        pop_skewness = pf.calculate_ela_distribution(X, y_hat)['ela_distr.skewness']
        tmp_meta = pf.calculate_ela_meta(X, y_hat)
        pop_fd_std = mf.calculate_fitness_distance_correlation(X, y_hat)['fitness_distance.fitness_std']
        tmp_nbc = pf.calculate_nbc(X, y_hat)
        data[['cma_skewness', 'cma_quad_simp_r2', 'cma_fd', 'cma_lin_inter_r2', 'cma_nbc_sd', 'cma_nbc_cor', 'cma_lin_simple_r2', 'cma_quad_inter_r2']] = np.array([pop_skewness, tmp_meta['ela_meta.quad_simple.adj_r2'], pop_fd_std,\
            tmp_meta['ela_meta.lin_w_interact.adj_r2'], tmp_nbc['nbc.nn_nb.sd_ratio'], tmp_nbc['nbc.nb_fitness.cor'], tmp_meta['ela_meta.lin_simple.adj_r2'], tmp_meta['ela_meta.quad_w_interact.adj_r2']])

        results.append(data)


    # Consolidate list of potentially multiple dataframes to a single dataframe
    df = pd.concat(results)
   
    return df


# Function wrapper needed for multiprocessing
if __name__ ==  '__main__':
    if len(sys.argv) == 1:

        fids = range(1, 25)
        dims = [2,3]
        iids = range(1, 2)

        cart_prod = itertools.product(*[fids, dims, iids])

        # Debug code:
        #for experiment in cart_prod:
        #    run_experiment(experiment)
        
        with Pool(36) as p:
            results = p.map(run_experiment, cart_prod)

        # Consolidate the results of each parallelized process into a single data frame
        data = pd.concat(results).reset_index(drop = True)
        data.to_csv(os.path.join(ROOT, f'00_ela_fg_results.csv'), index = False)
        
    else:
        raise SyntaxError("Insufficient number of arguments passed")
