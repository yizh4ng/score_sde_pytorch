import itertools

import numpy as np
import torch
import os
from tqdm import tqdm

from mog_util.misc import estimate_marginal_accuracy
from mog_util.reverse_step import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

reverse_fucs = ['alg1', 'alg1mala']
# step_sizes = [10, 40, 100, 200, 500]
total_steps = [2, 5, 10, 25]
mcmc_steps  = [20]
mcmc_step_sizes_scale = [0.1, 1, 10]

reverse_fucs = ['ddpm']
# step_sizes = [10, 40, 100, 200, 500]
total_steps = [42, 105, 210, 525]
mcmc_steps  = [None]
mcmc_step_sizes_scale = [None]

# parameters_combinations = list(itertools.product(reverse_fucs, step_sizes, mcmc_steps, mcmc_step_sizes_scale))
parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale))


# for step_size in [40, 100, 200, 500]:
for parameters in parameters_combinations:
    # reverse_fuc, step_size, mcmc_steps, mcmc_step_size_scale = parameters
    reverse_fuc, total_step, mcmc_steps, mcmc_step_size_scale = parameters
    reverse_fuc = reverse_step_dict[reverse_fuc]
    x = torch.randn(50000).to('cuda')
    pbar = tqdm(total=1000, leave=False)
    for t in reversed(np.append(np.linspace(0, 1000, total_step, endpoint=False)[1:], 1000)):
        x = reverse_fuc(x, t, step_size=1000/total_step, mcmc_steps=mcmc_steps, mcmc_step_size_scale=mcmc_step_size_scale)
        pbar.update(1000/total_step)
    # for t in reversed(range(1, 1001)):
    #     if t % total_step != 0: continue
    #     print(t)
    #     x = reverse_fuc(x, t, step_size=total_step, mcmc_steps=mcmc_steps, mcmc_step_size_scale=mcmc_step_size_scale)
    #     # x = reverse_ode_step(x, t, step_size=step_size)
    #     # x = reverse_sde_step(x, t, step_size=step_size)
    #     # x = reverse_ddim_step(x, t, step_size=step_size)
    #     # x = reverse_ddpm_step(x, t, step_size=step_size)
    #     # x = langevin_correction_explicit(x, t, step_size=step_size)
    #     # x = langevin_correction_alg1(x, t, step_size=step_size)
    #     # x = langevin_correction_with_rejection_alg1(x, t, step_size=step_size)
    #
    #     # x = langevin_correction(x, t, step_size=step_size)
    #     # x = mala(x, t, step_size=step_size)
    #     pbar.update(total_step)
    pbar.close()


    # kl = calculate_kl(y, x)
    mc = estimate_marginal_accuracy(y[:, None], x[:, None])

    if parameters[0] == 'ddpm':
        print(f'{parameters[0]}, total_steps: {total_step}:'
              f' marginal accuracy: {mc:.6f}, nfe: {total_step}')
    else:
        print(f'{parameters[0]}, total_steps: {total_step}, mcmc_steps: {mcmc_steps}, mcmc_step_size_scale: {mcmc_step_size_scale},'
              f' marginal accuracy: {mc:.6f}, nfe: {total_step * (mcmc_steps + 1)}')


    # Visualize above calculated histogram as bar diagram
    # hist = torch.histc(x, bins = 50, min = -5, max = 5).to('cpu')
    # import matplotlib.pyplot as plt
    # bins = range(50)
    # plt.bar(bins, hist, align='center')
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.title(f'kl div: {mc:.6f}, step szie: {step_size}')
    # plt.savefig(f'./test/{step_size}_sde.png')
    # # plt.show()
    # plt.close()

    # hist = torch.histc(y, bins = 50, min = -5, max = 5).to('cpu')
    # import matplotlib.pyplot as plt
    # bins = range(50)
    # plt.bar(bins, hist, align='center')
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.show()
    # plt.close()
