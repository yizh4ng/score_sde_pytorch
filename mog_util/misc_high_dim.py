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
        # _min = torch.min(torch.cat((samples_mu[:, i], samples_pi[:,i]))).item()
        # _max = torch.max(torch.cat((samples_mu[:, i], samples_pi[:,i]))).item()
        _min = -5
        _max = 5
        mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=_min,
                              max=_max)
        pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=_min,
                              max=_max)
        # mu_hist = torch.histc(samples_mu[:, i], bins=num_bins, min=-5.0,
        #                       max=5.0)
        # pi_hist = torch.histc(samples_pi[:, i], bins=num_bins, min=-5.0,
        #                       max=5.0)

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
