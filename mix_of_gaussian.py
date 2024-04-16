import numpy as np
import torch
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def grad_log_p(x_t, t,
               mu1=torch.tensor(1).to('cuda'),
               mu2=torch.tensor(-1).to('cuda'),
               sigma=torch.tensor(0.3).to('cuda'),
               beta=torch.tensor(0.01).to('cuda')):
    mu1_t = mu1 * torch.exp(-0.5 * beta * t)
    mu2_t = mu2 * torch.exp(-0.5 * beta * t)
    sigma_t = sigma**2 * torch.exp(-beta * t) + 1 - torch.exp(-beta * t)

    # 计算两个高斯成分的概率密度
    p1 = torch.exp(-0.5 * (x_t - mu1_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)
    p2 = torch.exp(-0.5 * (x_t - mu2_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)

    # 计算梯度对数似然
    grad_log_p = (-(x_t - mu1_t) * p1 - (x_t - mu2_t) * p2) / (p1 + p2)
    grad_log_p /= sigma_t

    return grad_log_p

def reverse_sde_step(x, t, step_size=1, beta=torch.tensor(0.01).to('cuda')):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size + torch.sqrt(beta * step_size) * torch.randn_like(x)

def reverse_ode_step(x, t, step_size=1, beta=torch.tensor(0.01)):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size

def reverse_ddim_step(x, t, step_size=1, beta=torch.tensor(0.01)):
    alpha = 1 - beta
    alpha_cumprod = alpha ** t
    alpha_cumrpod_prev = alpha ** (t - step_size)

    predicted_noise = -grad_log_p(x, t) * torch.sqrt(1 -alpha_cumprod)
    x0_t = (x - (predicted_noise * torch.sqrt((1 - alpha_cumprod)))) / torch.sqrt(alpha_cumprod)
    # c1 = 0 * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
    c1 = 0
    c2 = torch.sqrt((1 - alpha_cumrpod_prev) - c1 ** 2)
    x = torch.sqrt(alpha_cumrpod_prev) * x0_t + c2 * predicted_noise
    return x

def langevin_correction(x, t, step_size=1, beta=torch.tensor(0.01)):
    # record current x
    current_x = x

    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')

    # one step ddim as initial state for langevin mcmc
    x = reverse_sde_step(x, t, step_size, beta)
    # x = reverse_ddim_step(x, t, step_size, beta)
    # x = reverse_ode_step(x, t, step_size, beta)

    # avoid final redundant mcmc step
    if t - step_size < 1e-8:
        return x

    # set 10 step langevin iteration
    for _ in range(1000):
        # Calculate Score Components
        weight = 4 * beta * h  # 0.0011 -> 2e-05
        score = grad_log_p(x, t - step_size, beta=beta)
        term_1 = -(-2 * current_x * torch.exp(-h)) / weight
        term_2 = -(2 * x * torch.exp(-2 * h)) / weight

        # Calculate Score
        grad = score + term_1 + term_2
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        inner_step_size = torch.tensor(5e-4)
        x_mean = x + inner_step_size * grad
        x = x_mean + torch.sqrt(inner_step_size * 2) * noise
    return x


def langevin_correction_explicit(x, t, step_size=1, beta=torch.tensor(0.01)):
    # record current x
    current_x = x

    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')

    # one step ddim as initial state for langevin mcmc
    x = reverse_ddim_step(x, t, step_size, beta)

    # avoid final redundant mcmc step
    if t - step_size < 1e-8:
        return x

    # set 10 step langevin iteration
    for _ in range(100):
        # Calculate Score Components
        score = grad_log_p(x, t - step_size, beta=beta)
        noise = torch.randn_like(x).to('cuda')

        # update x
        inner_step_size = torch.tensor(5e-4)
        a = 1/ ( beta * (torch.exp(2 * h) - 1))
        term_1 = score * (torch.exp(a * inner_step_size)/a - 1/a)
        term_2 = torch.exp(h) * (torch.exp(a * inner_step_size) - 1) * current_x
        term_3 = noise * torch.sqrt((torch.exp(2 * a * inner_step_size) - 1) / a)
        x =  (x + term_1 + term_2 + term_3) / torch.exp(a * inner_step_size)
    return x

x = torch.randn(50000).to('cuda')

for step_size in [40]:
    pbar = tqdm(total=1000)
    for t in reversed(range(1, 1000)):
        if t % step_size != 0: continue
        # x = reverse_ode_step(x, t, step_size=step_size)
        # x = reverse_sde_step(x, t, step_size=step_size)
        # x = reverse_ddim_step(x, t, step_size=step_size)
        # x = langevin_correction(x, t, step_size=step_size)
        x = langevin_correction_explicit(x, t, step_size=step_size)
        pbar.update(step_size)

    # Visualize above calculated histogram as bar diagram
    hist = torch.histc(x, bins = 50, min = -5, max = 5).to('cpu')
    import matplotlib.pyplot as plt
    bins = range(50)
    plt.bar(bins, hist, align='center')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.show()

    y = torch.concat([torch.randn(25000) * 0.3 + 1, torch.randn(25000) * 0.3 - 1]).to('cuda')
    hist = torch.histc(y, bins = 50, min = -5, max = 5).to('cpu')
    import matplotlib.pyplot as plt
    bins = range(50)
    plt.bar(bins, hist, align='center')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.show()

    def calculate_kl(x, y):
        import torch
        from torch.distributions import Categorical
        from torch.distributions.kl import kl_divergence
        num_classes = 50
        def estimate_probs(data):
            # 计算每个类别的频率
            counts = torch.histc(data.float(), bins=num_classes, min=-5, max=5)
            # 将频率转换为概率
            probs = counts / counts.sum() + 1e-8
            return probs

        # 假设你有两组离散的数据

        # 估计概率分布
        probs1 = estimate_probs(x)
        probs2 = estimate_probs(y)

        dist1 = Categorical(probs=probs1)
        dist2 = Categorical(probs=probs2)

        # 计算KL散度
        kl = kl_divergence(dist1, dist2)
        print(f"KL divergence from dist1 to dist2: {kl.item()}")

    calculate_kl(y, x)