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
        # mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=-5.0,
        #                       max=5.0)
        # pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=-5.0,
        #                       max=5.0)
        mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=-1.5,
                              max=1.5)
        pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=-1.5,
                              max=1.5)

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

def calculate_wasserstein_distance(x, y):
    import ot
    cost_matrix = ot.dist(x[:, :2], y[:, :2]).to('cuda')

    # Regularization parameter
    # reg = 1e-5
    reg = 1e-1

    # Compute the optimal transport plan using Sinkhorn algorithm
    transport_plan = ot.sinkhorn(torch.ones(len(x)).to('cuda') / len(x),
                                 torch.ones(len(y)).to('cuda') / len(y),
                                 cost_matrix, reg)
    # transport_plan = ot.sinkhorn_unbalanced(torch.ones(len(x)).to('cuda') / len(x),
    #                              torch.ones(len(y)).to('cuda') / len(y),
    #                              cost_matrix, reg, reg)
    # transport_plan = ot.smooth.smooth_ot_dual(torch.ones(len(x)).to('cuda') / len(x),
    #                              torch.ones(len(y)).to('cuda') / len(y), cost_matrix, reg)

    # transport_plan = ot.emd(torch.ones(len(x)).to('cuda') / len(x),
    #                              torch.ones(len(y)).to('cuda') / len(y),
    #                              cost_matrix)

    # Calculate the Wasserstein distance (cost)
    wasserstein_distance = torch.sum(transport_plan * cost_matrix)
    return wasserstein_distance

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

def visualize_cluster(data, ground_truth, title, mode=['hist', 'cluster'], alpha=0.01):
    c = 'grey'
    label = None
    if 'DDPM' in title:
        c = 'blue'
        label = 'DDPM'
    if 'MALA' in title:
        c = 'red'
        label = 'MALA'
    if 'MALA_ES' in title:
        c = 'brown'
        label = 'MALA_ES'
    if 'Langevin' in title:
        title = 'ULA'
        c = 'orange'
        label = 'ULA'
    if 'ULD' in title:
        c = 'green'
        label = 'ULD'
    if 'ALD' in title:
        c = 'purple'
        label = 'ALD'
    if 'DDIM' in title:
        c = 'black'
        label = 'DDIM'

    if 'cluster' in mode:
        import matplotlib.pyplot as plt
        _data = data.cpu().numpy()
        np.random.shuffle(_data)
        plt.figure(figsize=(6, 6))  # Set the figure size
        scatter = plt.scatter(_data[:50000, 0], _data[:50000, 1], alpha=alpha,
                              # cmap='viridis',
                              c=c,
                              s=10)  # s is the size of points
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.xlim(-1.75, 1.75)
        plt.ylim(-1.75, 1.75)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{title}.png', dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
    if 'hist' in mode:
        # data = data.cpu().numpy()
        data = data[(data[:,0] < -0.75) & (data[:, 1] < 0.5) & (data[:, 1] > -0.5)]
        hist = torch.histc(data[:, 1], bins = 50, min = -0.5, max = 0.5).to('cpu')
        hist = hist / hist.sum()
        import matplotlib.pyplot as plt
        bins = np.linspace(-0.5, 0.5, 50, endpoint=False)
        bar_width = bins[1] - bins[0]
        plt.bar(bins, hist,
                align='edge',
                alpha=1, color=c, width=bar_width, label=label)

        # _ground_truth = ground_truth[(ground_truth[:,0] < -0.75) & (ground_truth[:, 1] < 0.5) & (ground_truth[:, 1] > -0.5)]
        # ground_truth_hist = torch.histc(_ground_truth[:, 1], bins = 50, min = -0.5, max = 0.5).to('cpu')
        # ground_truth_hist = ground_truth_hist / ground_truth_hist.sum()
        # hist = np.histogram(data[:, 1], bins = 50, range = (-0.5, 0.5))
        # plt.bar(bins, ground_truth_hist, align='center', alpha=0.5, color='grey', width=bar_width, label='Ground Truth')

        import scipy.stats as stats
        weights = (1/3 / 50,) * 3
        means = (-3.4202e-01, 0, 3.4202e-01)
        stds = (0.007 ** 0.5, ) * 3
        # stds = (0.01 ** 0.5, ) * 3
        x = np.linspace(-0.5, 0.5, 100)
        pdf = np.zeros_like(x)
        for weight, mean, std in zip(weights, means, stds):
            pdf += weight * stats.norm.pdf(x, mean, std)
        plt.plot(x, pdf, label='Ground Truth', color='grey', linewidth=3)

        plt.xlabel('Dimension 2')
        plt.ylabel('Probability')
        # plt.title(f'kl div: {mc:.6f}, step szie: {step_size}')
        plt.title(title)
        plt.legend(loc='upper right')
        # plt.savefig(f'./test/{step_size}_sde.png')
        plt.savefig(f'{title}_dim.png', dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
