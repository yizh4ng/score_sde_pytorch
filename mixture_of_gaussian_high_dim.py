import torch
import numpy as np
import itertools
from tqdm import tqdm

from mog_util.misc_high_dim import estimate_marginal_accuracy
from mog_util.reverse_step_high_dim import *
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")

reverse_fucs = ['alg1', 'alg1mala']
# reverse_fucs = ['alg1mala']
total_steps = [2, 5, 10, 25]
# total_steps = [2, 5]
# total_steps = [25]
mcmc_steps  = [20]
mcmc_step_sizes_scale = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
# mcmc_step_sizes_scale = [0.1, 1, 10]
inits = ['ddim', 'ddpm', 'ddpm_drift']
# inits = ['ddim', 'ddpm']
# inits = [None]
weight_scale = [1]
#
# reverse_fucs = ['ddpm']
# # total_steps = [42, 105, 210, 525]
# total_steps = [40, 100, 200, 500]
# mcmc_steps  = [None]
# mcmc_step_sizes_scale = [None]

parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale,
                                                 inits, weight_scale))

import pandas as pd

df = pd.DataFrame(columns=['reverse_fuc', 'total_step', 'mcmc_step', 'mcmc_step_size_scale', 'inits', 'weight_scale', 'ma'])

for parameters in parameters_combinations:
    reverse_fuc, total_step, mcmc_step, mcmc_step_size_scale, init, weight_scale = parameters
    reverse_fuc = reverse_step_dict[reverse_fuc]
    x = torch.randn([50000, d]).to('cuda')
    pbar = tqdm(total=1000, leave=False)
    for t in reversed(np.append(np.linspace(0, 1000, total_step, endpoint=False)[1:], 1000)):
        x = reverse_fuc(x, t, step_size=1000/total_step, mcmc_steps=mcmc_step,
                        mcmc_step_size_scale=mcmc_step_size_scale, init=init, weight_scale=weight_scale)
        pbar.update(1000/total_step)
    pbar.close()

    data = x.cpu().numpy()
    # Visual
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 8))  # Set the figure size
    # scatter = plt.scatter(data[:, 0], data[:, 1],  alpha=0.5, cmap='viridis',
    #                       s=10)  # s is the size of points
    # plt.colorbar(scatter)  # Show color scale
    # plt.title('Cluster Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.show()    # y = torch.concat([torch.randn(25000) * 0.3 + 1, torch.randn(25000) * 0.3 - 1]).to('cuda')

    # data = samples.cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 8))  # Set the figure size
    # scatter = plt.scatter(data[:, 0], data[:, 1],  alpha=0.5, cmap='viridis',
    #                       s=10)  # s is the size of points
    # plt.colorbar(scatter)  # Show color scale
    # plt.title('Cluster Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.show()    # y = torch.concat([torch.randn(25000) * 0.3 + 1, torch.randn(25000) * 0.3 - 1]).to('cuda')
    mc = estimate_marginal_accuracy(y, x)

    if parameters[0] == 'ddpm':
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
    for total_step in total_steps:
        filtered_df = df[(df['total_step'] == total_step) & (df['reverse_fuc'] == reverse_fuc)]
        best_performance = filtered_df[filtered_df['ma'] == filtered_df['ma'].max()]
        print(best_performance.values[0])