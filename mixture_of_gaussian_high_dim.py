import torch
import numpy as np
import itertools
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from mog_util.misc import visualize_cluster, calculate_wasserstein_distance
from mog_util.misc_high_dim import estimate_marginal_accuracy
from mog_util.reverse_step_high_dim import *
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")
warnings.filterwarnings("ignore", message="clamping frac")


reverse_fucs = ['Langevin', 'MALA']
# reverse_fucs = ['alg1mala']
# total_steps = [1]
total_steps = list(range(1, 6, 1))
mcmc_steps = [20]
mcmc_step_sizes_scale = [1]
inits = ['ddpm']
# inits = ['ddim', 'ddpm', 'ddpm_drift']
# inits = [None]
weight_scale = [1]
#
reverse_fucs = ['DDPM']
# total_steps = [40, 100, 200, 500]
# total_steps = [42, 105, 210, 525]
total_steps, vis = list(range(25, 525, 20)), False
# total_steps, vis = list(range(1, 1+25*1, 1)), False
# total_steps, vis = list(range(50, 1000, 20)), False
# total_steps, vis = [2000], True
# total_steps, vis = [205], True
# total_steps, vis = [1000], True
mcmc_steps  = [None]
mcmc_step_sizes_scale = [None]
inits = [None]
weight_scale = [None]
# visualize = lambda _x, y, _title : visualize_cluster(_x, y, _title, ['cluster', 'hist'], alpha=0.01)
visualize = lambda _x, y, _title : visualize_cluster(_x, y, _title, ['cluster'], alpha=0.01)

parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale,
                                                 inits, weight_scale))

import pandas as pd

df = pd.DataFrame(columns=['reverse_fuc', 'total_step', 'mcmc_step', 'mcmc_step_size_scale', 'inits', 'weight_scale', 'ma'])


for parameters in parameters_combinations:
    reverse_fuc, total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale = parameters
    reverse_fuc = reverse_step_dict[reverse_fuc]
    x = torch.randn([5000, d]).to('cuda')
    pbar = tqdm(total=1000, leave=False)
    for t in reversed(np.append(np.linspace(0, 1000, total_step, endpoint=False)[1:], 1000)):
    # for t in reversed(np.linspace(0, , total_step, endpoint=False)[1:]):
    # for t in reversed(linespace(0, 1000, total_step)[1:]):
        x = reverse_fuc(x, t, step_size=1000/total_step, mcmc_steps=mcmc_step,
                    mcmc_step_size_scale=mcmc_step_size_scale, init=init, weight_scale=weight_scale)
        pbar.update(1000/total_step)
    pbar.close()

    if vis:
        visualize(x, y, f'{parameters[0]}')


    # mc = estimate_marginal_accuracy(y, x, num_bins=200)
    mc = calculate_wasserstein_distance(x,y)


    if parameters[0] in ['DDPM', 'DDIM', 'SDE', 'ODE']:
        df.loc[len(df)] = [parameters[0], total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale, mc.to('cpu').numpy()]
        print(f'{parameters[0]}, total_steps: {total_step}:'
              f' marginal accuracy: {mc:.6f}, nfe: {total_step}')
    else:
        df.loc[len(df)] = [parameters[0], total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale, mc.to('cpu').numpy()]
        # df = df.ap({'total_step':total_step, 'mcmc_step':mcmc_step, 'mcmc_step_size_scale':mcmc_step_size_scale,
        #            'inits':init, 'weight_scale':weight_scale, 'ma':mc})
        print(f'{parameters[0]}, total_step: {total_step}, mcmc_steps: {mcmc_step}, mcmc_step_size_scale: {mcmc_step_size_scale},'
              f' init: {init}, marginal accuracy: {mc:.6f}, nfe: {total_step * (mcmc_step + 1)}')

print(f'Best performance:')
for reverse_fuc in reverse_fucs:
    print(reverse_fuc)
    for total_step in total_steps:
        filtered_df = df[(df['total_step'] == total_step) & (df['reverse_fuc'] == reverse_fuc)]
        best_performance = filtered_df[filtered_df['ma'] == filtered_df['ma'].max()]
        print(float(best_performance['ma'].values))
        # print(best_performance.values)

# if vis:
#     visualize(y, 'Ground Truth Samples')
