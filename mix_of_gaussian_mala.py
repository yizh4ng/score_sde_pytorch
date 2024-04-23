import torch
from tqdm import tqdm

# 设定目标分布参数
mu1, sigma1 = torch.tensor(1.0).to('cuda'), torch.tensor(0.3).to('cuda')
mu2, sigma2 = torch.tensor(-1.0).to('cuda'), torch.tensor(0.3).to('cuda')

# 目标概率密度函数
def log_prob(x):
    # 计算两个高斯成分的概率密度
    p1 = torch.exp(-0.5 * (x - mu1)**2 / sigma1**2) / torch.sqrt(2 * torch.pi * sigma1**2)
    p2 = torch.exp(-0.5 * (x - mu2)**2 / sigma2**2) / torch.sqrt(2 * torch.pi * sigma2**2)
    return torch.log(0.5 * p1 + 0.5 * p2)

# 解析梯度
def grad_log_prob(x):
    # 计算两个高斯成分的概率密度
    p1 = torch.exp(-0.5 * (x - mu1)**2 / sigma1**2) / torch.sqrt(2 * torch.pi * sigma1**2)
    p2 = torch.exp(-0.5 * (x - mu2)**2 / sigma2**2) / torch.sqrt(2 * torch.pi * sigma2**2)

    # 计算梯度对数似然
    grad_log_p = (-(x - mu1) * p1 - (x - mu2) * p2) / (p1 + p2)
    # assume sigma1 == sigma2
    grad_log_p /= sigma1**2
    return grad_log_p

# MALA采样器
def mala(num_samples, num_independent_samples, x_init, step_size):
    x = x_init
    pbar = tqdm(total=num_samples)
    for _ in range(num_samples - 1):  # 只保存最后一个样本
        x_new = x + step_size * grad_log_prob(x) + (2.0 * step_size) ** 0.5 * torch.randn_like(x).to('cuda')
        accept_ratio = torch.exp(log_prob(x_new) - log_prob(x) - 0.5 * ((x_new - x - step_size * grad_log_prob(x)) ** 2) / (2 * step_size) + 0.5 * ((x - x_new - step_size * grad_log_prob(x_new)) ** 2) / (2 * step_size))
        accept = torch.rand_like(x).to('cuda') < accept_ratio
        x = torch.where(accept, x_new, x)
        # x = x + step_size * grad_log_prob(x) + (2.0 * step_size) ** 0.5 * torch.randn_like(x).to('cuda')
        pbar.update(1)
    pbar.close()
    return x

# 参数设置
num_samples = 10 # 迭代次数
num_independent_samples = 50000 # 独立样本数量
x_init = torch.zeros(num_independent_samples).to('cuda')  # 初始化
# x_init = torch.rand(num_independent_samples).to('cuda') * 2 -1 # 初始化
step_size = 0.1  # 梯度步长

# 运行MALA
final_samples = mala(num_samples, num_independent_samples, x_init, step_size)
print("Final samples shape:", final_samples.shape)  # 应该是 (num_independent_samples,)

import matplotlib.pyplot as plt
# 可视化结果
plt.figure(figsize=(10, 6))
plt.hist(final_samples.to('cpu').numpy(), bins=100, alpha=0.75, density=True, label='MALA Samples')
plt.title('Histogram of MALA Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

y = torch.concat([torch.randn(25000) * 0.3 + 1, torch.randn(25000) * 0.3 - 1]).to('cuda')
# hist = torch.histc(y, bins=50, min=-5, max=5).to('cpu')
# import matplotlib.pyplot as plt
#
# bins = range(50)
# plt.bar(bins, hist, align='center')
# plt.xlabel('Bins')
# plt.ylabel('Frequency')
# plt.show()
#

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


calculate_kl(y, final_samples)
