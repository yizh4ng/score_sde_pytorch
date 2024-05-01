import torch


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
