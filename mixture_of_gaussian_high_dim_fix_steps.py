import torch
import numpy as np
import itertools
from tqdm import tqdm

from mog_util.misc_high_dim import estimate_marginal_accuracy
from mog_util.reverse_step_high_dim import *
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")
warnings.filterwarnings("ignore", message="clamping frac")


reverse_fucs = ['Langevin', 'MALA']
# reverse_fucs = ['alg1mala']
# total_steps = [1]
total_steps = [5]
mcmc_steps  = list(range(2, 26 * 20, 20))
# mcmc_step_sizes_scale = [0.6, 0.8, 1, 1.5, 3]
mcmc_step_sizes_scale = [1,2,4,8]
inits = ['ddpm']
weight_scale = [1]
#
parameters_combinations = list(itertools.product(reverse_fucs, total_steps, mcmc_steps, mcmc_step_sizes_scale,
                                                 inits, weight_scale))

import pandas as pd

df = pd.DataFrame(columns=['reverse_fuc', 'total_step', 'mcmc_step', 'mcmc_step_size_scale', 'inits', 'weight_scale', 'ma'])

def vis(data, title):
    import matplotlib.pyplot as plt
    data = data.cpu().numpy()
    np.random.shuffle(data)
    plt.figure(figsize=(6, 6))  # Set the figure size
    scatter = plt.scatter(data[:50000, 0], data[:50000, 1], alpha=0.05, cmap='viridis',
                          s=10)  # s is the size of points
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

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

    # vis(x, f'{parameters[0]}')

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

# vis(y, 'Ground Truth Samples')
