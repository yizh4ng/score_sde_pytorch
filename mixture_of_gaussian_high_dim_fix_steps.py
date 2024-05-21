import torch
import numpy as np
import itertools
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from mog_util.misc import visualize_cluster
from mog_util.misc_high_dim import estimate_marginal_accuracy
from mog_util.reverse_step_high_dim import *
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")
warnings.filterwarnings("ignore", message="clamping frac")


# reverse_fucs = ['Langevin', 'MALA']
# reverse_fucs, mcmc_step_sizes_scale = ['Langevin'],  [0.2]
# reverse_fucs, mcmc_step_sizes_scale = ['MALA'],  [0.8]
reverse_fucs, mcmc_step_sizes_scale, vis = ['Langevin'],  [0.3], True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['Langevin'],  np.linspace(0.1, 1, 10, endpoint=True), True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['Langevin'],  np.linspace(1, 2, 10, endpoint=True), False

reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA'],  [0.2], True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA'],  np.linspace(0.01, 0.1, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA'],  np.linspace(0.1, 1, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA'],  np.linspace(1, 2, 10, endpoint=True), False

reverse_fucs, mcmc_step_sizes_scale, vis = ['ULD'],  [0.005], True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['uld'],  np.linspace(0.01, 0.1, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['uld'],  np.linspace(0.001, 0.01, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['uld'],  np.linspace(0.0001, 0.001, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['uld'],  np.linspace(0.00001, 0.0001, 10, endpoint=True), False
# reverse_fucs, mcmc_step_sizes_scale, vis = ['uld'],  np.linspace(0.1, 1, 10, endpoint=True), False

reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA_ES'],  [0.2], True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA'],  [0.2], True
# reverse_fucs, mcmc_step_sizes_scale, vis = ['MALA_ES'],  np.linspace(0.1, 1, 10, endpoint=True), False

total_steps = [5]
# total_steps = [8]
# mcmc_steps, vis  = list(range(20, 26 * 20, 20)), False
mcmc_steps= [205]
# mcmc_steps= [525]
# mcmc_step_sizes_scale = [0.5]
# mcmc_step_sizes_scale = [1,2,3, 4, 5,6,7,8]
# mcmc_step_sizes_scale = [0.01, 0.03,0.1,0.3,1,2,3, 4, 5,6,7,8]
inits = ['ddpm']
weight_scale = [1]
visualize = lambda _x, _y, _title : visualize_cluster(_x,_y, _title, ['hist', 'cluster'], alpha=0.01)
# visualize = lambda _x, _title : visualize_cluster(_x, _title, ['hist'], alpha=0.01)
#
parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale,
                                                 inits, weight_scale))

import pandas as pd

df = pd.DataFrame(columns=['reverse_fuc', 'total_step', 'mcmc_step', 'mcmc_step_size_scale', 'inits', 'weight_scale', 'ma'])

for parameters in parameters_combinations:
    reverse_fuc, total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale = parameters
    reverse_fuc = reverse_step_dict[reverse_fuc]
    x = torch.randn([50000, d]).to('cuda')
    pbar = tqdm(total=1000, leave=False)
    real_mcmc_step = int(mcmc_step / total_step)
    for t in reversed(np.append(np.linspace(0, 1000, total_step, endpoint=False)[1:], 1000)):
    # for t in reversed(linespace(0, 1000, total_step)[1:]):
        x = reverse_fuc(x, t, step_size=1000/total_step, mcmc_steps=real_mcmc_step,
                    mcmc_step_size_scale=mcmc_step_size_scale, init=init, weight_scale=weight_scale)
        pbar.update(1000/total_step)
    pbar.close()

    if vis:
        # visualize(x, f'{parameters[0]}_{mcmc_step_size_scale}')
        visualize(x, y, f'{parameters[0]}')

    mc = estimate_marginal_accuracy(y, x, num_bins=200)


    if parameters[0] == 'DDPM':
        df.loc[len(df)] = [parameters[0], total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale, mc.to('cpu').numpy()]
        print(f'{parameters[0]}, total_steps: {total_step}:'
              f' marginal accuracy: {mc:.6f}, nfe: {total_step}')
    else:
        df.loc[len(df)] = [parameters[0], total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale, mc.to('cpu').numpy()]
        # df = df.ap({'total_step':total_step, 'mcmc_step':mcmc_step, 'mcmc_step_size_scale':mcmc_step_size_scale,
        #            'inits':init, 'weight_scale':weight_scale, 'ma':mc})
        print(f'{parameters[0]}, total_step: {total_step}, mcmc_steps: {mcmc_step}, mcmc_step_size_scale: {mcmc_step_size_scale},'
              f' init: {init}, marginal accuracy: {mc:.6f}, nfe: {total_step * (real_mcmc_step + 1)}')

print(f'Best performance:')
for reverse_fuc in reverse_fucs:
    print(reverse_fuc)
    for mcmc_step in mcmc_steps:
        filtered_df = df[(df['mcmc_step'] == mcmc_step) & (df['reverse_fuc'] == reverse_fuc)]
        best_performance = filtered_df[filtered_df['ma'] == filtered_df['ma'].max()]
        print(float(best_performance['ma'].values))
        # print(best_performance.values)

# if vis:
#     visualize(y, 'Ground Truth Samples')
