# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# How to install dependencies:
# 1. mamba env create -f environment.yml
# 2. conda activate ela_func_gen
# 3. git clone git@github.com:numbbo/coco.git
# 4. cd coco/
# 5. python do.py run-python

import cocoex

import nevergrad as ng
from nevergrad import functions as ngfuncs
from nevergrad.benchmark import registry as xpregistry
from nevergrad.benchmark import Experiment

import cma

import torch
from torch import nn

import numpy as np

from pathlib import Path

BASE_PATH = Path(".")
MODEL_PATH = Path("..data/3d/models/simple_tanh")


# this file implements:
# - an additional test function: CustomFunction
# - an addition experiment plan: additional_experiment
# it can be used with the --imports parameters if nevergrad.benchmark commandline function


class BBOBFunction(ngfuncs.ExperimentFunction):
    """Make Coco problem suite benchmarkable for Nevergrad"""

    def __init__(self, func, iid, dim):
        super().__init__(self.oracle_call, ng.p.Array(
            shape=[2], lower=[-5, -5], upper=[5, 5]).set_name(""))

        self.add_descriptors(
            func=func,
            iid=iid,
            dim=dim

        )
        
        self.suite = cocoex.Suite(
            "bbob", f"instances:{iid}", f"function_indices:1-24 dimensions:{dim}")
        self.func_index = func - 1

    def oracle_call(self, x):  # np.ndarray as input
        """Implements the call of the function."""
        f = self.suite[self.func_index]
        return f(x)


class NNFunction(ngfuncs.ExperimentFunction):
    """Make our NN functions benchmarkable for Nevergrad"""

    def __init__(self, func, rep, dim):
        super().__init__(self.oracle_call, ng.p.Array(
            shape=[3], lower=[-5, -5, -5],  upper=[5, 5, 5]).set_name(""))

        self.add_descriptors(
            func=func,
            rep=rep,
            dim=dim
        )

        self.model = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.model.load_state_dict(torch.load(Path(
            MODEL_PATH / f"f{func}_d3_i1_cma_sample_NNrep_{rep}.pt"), map_location=torch.device('cpu')), strict=True)

    def oracle_call(self, x):  # np.ndarray as input
        """Implements the call of the function."""
        f = self.model.forward(torch.tensor(
            x, dtype=torch.float32)).detach().numpy()
        return f[0]



@xpregistry.register  # register experiments in the experiment registry
def additional_experiment():  # The signature can also include a seed argument if need be (see experiments.py)
    #reps = [9, 9, 8, 9, 'first', 4, 3, 2, 'first', 2, 4, 7, 9, 8, 3, 2, 7, 2,
    #   3, 4, 6, 1, 1, 'first'] (2d)
    # for 3d (time issues) we use first nn reps of first point clouds (not reps sorted by euclid dist of ela vector)
    funcs = [NNFunction(func=fid, rep=rep, dim=3) for fid in range(5) for rep in range(1)]#[BBOBFunction(func=fid, iid=1, dim=2) for fid in range(1, 25)]# + [NNFunction(func=fid, rep=1, dim=3)
                                                                            #  for fid in range(1, 25)]

    nm_restart = ng.optimizers.NonObjectOptimizer(
        method="Nelder-Mead", random_restart=True).set_name("NelderMeadRandomStart")

    cobyla_restart = ng.optimizers.NonObjectOptimizer(
        method="COBYLA", random_restart=True).set_name("COBYLARandomStart")

    optimizers = [
        ng.optimizers.PSO,
        cobyla_restart,
        ng.optimizers.DifferentialEvolution(),
        ng.optimizers.EMNA(),
        ng.optimizers.DiagonalCMA,
        ng.optimizers.RandomSearch,
        nm_restart
    ]

    budgets = [10, 30, 100, 300, 1000, 3000, 10000, 30000]

    for budget in budgets:
        for optimizer in optimizers:
            for func in funcs:
                for seed in range(0xC0FFEE + 10, 0xC0FFEE + 20):
                    yield Experiment(func, optimizer=optimizer, budget=budget, num_workers=1, seed=seed)
