import torch
import numpy as np
import itertools
from tqdm import tqdm

from mog_util.misc_high_dim import estimate_marginal_accuracy
from mog_util.reverse_step_high_dim import *


# reverse_fucs = ['alg1', 'alg1mala']
# # step_sizes = [10, 40, 100, 200, 500]
# total_steps = [2, 5, 10, 25]
# mcmc_steps  = [20]
# mcmc_step_sizes_scale = [0.1, 1, 10]

reverse_fucs = ['ddpm']
# step_sizes = [10, 40, 100, 200, 500]
total_steps = [42, 105, 210, 525]
mcmc_steps  = [None]
mcmc_step_sizes_scale = [None]

parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale))


for parameters in parameters_combinations:
    reverse_fuc, total_step, mcmc_steps, mcmc_step_size_scale = parameters
    reverse_fuc = reverse_step_dict[reverse_fuc]
    x = torch.randn([50000, d]).to('cuda')
    pbar = tqdm(total=1000, leave=False)
    for t in reversed(np.append(np.linspace(0, 1000, total_step, endpoint=False)[1:], 1000)):
        x = reverse_fuc(x, t, step_size=1000/total_step, mcmc_steps=mcmc_steps, mcmc_step_size_scale=mcmc_step_size_scale)
        pbar.update(1000/total_step)
    # pbar = tqdm(total=1000)
    # for t in reversed(range(1, 1001)):
    #     if t % step_size != 0: continue
    #     # x = reverse_ode_step(x, t, step_size=step_size)
    #     # x = reverse_sde_step(x, t, step_size=step_size)
    #     # x = reverse_ddim_step(x, t, step_size=step_size)
    #     # x = reverse_ddpm_step(x, t, step_size=step_size)
    #     # x = langevin_correction_explicit(x, t, step_size=step_size)
    #     # x = langevin_correction_alg1(x, t, step_size=step_size)
    #     x = langevin_correction_with_rejection_alg1(x, t, step_size=step_size)
    #
    #     # x = langevin_correction(x, t, step_size=step_size)
    #     # x = mala(x, t, step_size=step_size)
    #     pbar.update(step_size)
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
        print(f'{parameters[0]}, total_steps: {total_step}, mcmc_steps: {mcmc_steps}, mcmc_step_size_scale: {mcmc_step_size_scale},'
              f' marginal accuracy: {mc:.6f}, nfe: {total_step * (mcmc_steps + 1)}')
