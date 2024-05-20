import torch
import numpy as np


def compute_tv_distance(p, q):
    # 计算总变异距离，假设p和q为离散分布的直方图
    return 0.5 * torch.sum(torch.abs(p - q))

def estimate_marginal_accuracy(samples_mu, samples_pi, num_bins=50):
    d = samples_mu.shape[1]  # 维度
    tv_distances = []

    # 为每个维度计算边缘分布的TV距离
    for i in range(d):
        # 计算每个维度的直方图
        # mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=float(samples_mu[:, i].min()),
        #                       max=float(samples_mu[:, i].max()))
        # pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=float(samples_pi[:, i].min()),
        #                       max=float(samples_pi[:, i].max()))
        mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=-5.0,
                              max=5.0)
        pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=-5.0,
                              max=5.0)

        # 归一化直方图
        mu_hist /= mu_hist.sum()
        pi_hist /= pi_hist.sum()

        # 计算TV距离
        tv_dist = compute_tv_distance(mu_hist, pi_hist)
        tv_distances.append(tv_dist)

    # 计算边缘精度
    marginal_acc = 1 - torch.mean(torch.stack(tv_distances)) / 2
    # print(f'{marginal_acc:.6f}')
    return marginal_acc

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
    print(f"KL divergence from dist1 to dist2: {kl.item():.6f}")
    return kl.item()


def visualize_hist(x, title, save_path=None):
    hist = torch.histc(x, bins = 50, min = -5, max = 5).to('cpu')
    import matplotlib.pyplot as plt
    bins = range(50)
    plt.bar(bins, hist, align='center')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    # plt.title(f'kl div: {mc:.6f}, step szie: {step_size}')
    plt.title(title)
    # plt.savefig(f'./test/{step_size}_sde.png')
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def visualize_cluster(data, title, mode=['hist', 'cluster'], alpha=0.01):
    if 'cluster' in mode:
        import matplotlib.pyplot as plt
        _data = data.cpu().numpy()
        np.random.shuffle(_data)
        plt.figure(figsize=(6, 6))  # Set the figure size
        scatter = plt.scatter(_data[:, 0], _data[:, 1], alpha=alpha, cmap='viridis',
                              s=10)  # s is the size of points
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.tight_layout()
        plt.savefig(f'{title}.png', dpi=600, bbox_inches='tight')
        # plt.show()
        plt.close()
    if 'hist' in mode:
        # data = data.cpu().numpy()
        data = data[(data[:,0] < -0.75) & (data[:, 1] < 0.5) & (data[:, 1] > -0.5)]
        hist = torch.histc(data[:, 1], bins = 50, min = -0.5, max = 0.5).to('cpu')
        # hist = np.histogram(data[:, 1], bins = 50, range = (-0.5, 0.5))
        import matplotlib.pyplot as plt
        bins = range(50)
        plt.bar(bins, hist, align='center')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        # plt.title(f'kl div: {mc:.6f}, step szie: {step_size}')
        plt.title(title)
        # plt.savefig(f'./test/{step_size}_sde.png')
        plt.savefig(f'{title}_dim.png', dpi=600, bbox_inches='tight')
        # plt.show()
        plt.close()
