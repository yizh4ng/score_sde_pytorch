import torch
from tqdm import tqdm

d = 10  # Dimensionality
pis = torch.tensor([0.5, 0.5]).to('cuda')
# mus = [torch.tensor([1.0, 1.0]).to('cuda'), torch.tensor([-1.0, -1.0]).to('cuda')]
mus = [torch.tensor((0.3,) * d).to('cuda'), torch.tensor((-0.3,) * d).to('cuda')]
sigmas = [0.3 * torch.eye(d).to('cuda'), 0.3 * torch.eye(d).to('cuda')]

def grad_log_p(x_t, t, mus=mus, sigmas=sigmas, pis=pis, beta=torch.tensor(0.01).to('cuda')):
    n_samples = x_t.shape[0]  # Number of samples in the batch
    d = x_t.shape[1]  # Dimensionality of each sample
    k = len(pis)  # Number of mixture components
    weighted_sum = torch.zeros_like(x_t)
    p_x_t = torch.zeros(n_samples).to('cuda')

    for i in range(k):
        mu_i = mus[i]
        sigma_i = sigmas[i]

        # Time-dependent mean and covariance
        mu_i_t = mu_i * torch.exp(-0.5 * beta * t)
        sigma_i_t = sigma_i * torch.exp(-beta * t) + (1 - torch.exp(-beta * t)) * torch.eye(d).to(x_t.device)

        # Inverse and determinant of covariance matrix
        sigma_i_t_inv = torch.inverse(sigma_i_t)
        det_sigma_i_t = torch.det(sigma_i_t)

        # Compute the Gaussian density
        diff = x_t - mu_i_t
        exp_component = torch.exp(-0.5 * torch.sum(diff @ sigma_i_t_inv * diff, dim=1))
        normal_density_i = exp_component / torch.sqrt((2 * torch.pi) ** d * det_sigma_i_t)

        # Weight the density by the mixture coefficients
        weighted_density_i = pis[i] * normal_density_i
        p_x_t += weighted_density_i

        # Accumulate the weighted sum for the gradient calculation
        weighted_sum += weighted_density_i.unsqueeze(1) * (sigma_i_t_inv @ diff.T).T

    # Compute the gradient of the log probability
    grad_log_p = -weighted_sum / p_x_t.unsqueeze(1)

    return grad_log_p
def log_p(x_t, t, mus=mus, sigmas=sigmas, pis=pis, beta=torch.tensor(0.01).to('cuda')):
    n_samples = x_t.shape[0]  # Number of samples in the batch
    d = x_t.shape[1]  # Dimensionality of each sample
    k = len(pis)  # Number of mixture components
    log_p_x_t = torch.zeros(n_samples, device=x_t.device)

    for i in range(k):
        mu_i = mus[i]
        sigma_i = sigmas[i]

        # Time-dependent mean and covariance
        mu_i_t = mu_i * torch.exp(-0.5 * beta * t)
        sigma_i_t = sigma_i * torch.exp(-beta * t) + (1 - torch.exp(-beta * t)) * torch.eye(d, device=x_t.device)

        # Inverse and determinant of the covariance matrix
        sigma_i_t_inv = torch.inverse(sigma_i_t)
        det_sigma_i_t = torch.det(sigma_i_t)

        # Compute the Gaussian density
        diff = x_t - mu_i_t
        exp_component = torch.exp(-0.5 * torch.sum(diff @ sigma_i_t_inv * diff, dim=1))
        normal_density_i = exp_component / torch.sqrt((2 * torch.pi) ** d * det_sigma_i_t)

        # Accumulate the log probability
        log_p_x_t += pis[i] * normal_density_i

    # Take the log of the accumulated probabilities
    log_p_x_t = torch.log(log_p_x_t)

    return log_p_x_t

def reverse_sde_step(x, t, step_size=1, beta=torch.tensor(0.01).to('cuda')):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size + torch.sqrt(beta * step_size) * torch.randn_like(x)

def reverse_ode_step(x, t, step_size=1, beta=torch.tensor(0.01)):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size
def reverse_ddim_step(x, t, step_size=1, beta=torch.tensor(0.01), eta=0):
    alpha = 1 - beta
    alpha_cumprod = alpha ** t
    alpha_cumrpod_prev = alpha ** (t - step_size)

    predicted_noise = -grad_log_p(x, t) * torch.sqrt(1 -alpha_cumprod)

    epsilon_t = torch.randn_like(x)
    sigma_t = eta * torch.sqrt((1 - alpha_cumrpod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumrpod_prev))

    x = (
            torch.sqrt(alpha_cumrpod_prev / alpha_cumprod) * x +
            (torch.sqrt(1 - alpha_cumrpod_prev - sigma_t ** 2) - torch.sqrt(
                (alpha_cumrpod_prev * (1 - alpha_cumprod)) / alpha_cumprod)) * predicted_noise +
            sigma_t * epsilon_t
    )
    return x

def reverse_ddpm_step(x, t, step_size=1, beta=torch.tensor(0.01)):
    x = reverse_ddim_step(x, t, step_size, beta, eta=1)
    return x
def langevin_correction(x, t, step_size=1, beta=torch.tensor(0.01)):
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    # x = reverse_ddim_step(x, t, step_size, beta)

    # set 10 step langevin iteration
    for _ in range(10):
        # Calculate Score Components
        score = grad_log_p(x, t - step_size, beta=beta)

        # Calculate Score
        grad = score
        # grad = score
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        inner_step_size = torch.tensor(0.5)
        x_mean = x + inner_step_size * grad
        x = x_mean + torch.sqrt(inner_step_size * 2) * noise
    return x

def mala(x, t, step_size=1, beta=torch.tensor(0.01)):
    # one step ddim as initial state for langevin mcmc
    # x = reverse_ddim_step(x, t, step_size, beta)

    # langevin iteration
    for _ in range(10):
        def get_grad(x, t):
            score = -grad_log_p(x, t, beta=beta)
            return score
        def get_energy(x, t):
            energy = -log_p(x, t, beta=beta)
            return energy

        # Calculate Score
        grad = get_grad(x, t - step_size)
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        inner_step_size = torch.tensor(0.5)
        x_mean = x - inner_step_size * grad
        x_new = x_mean + torch.sqrt(inner_step_size * 2) * noise

        # now decide whether to accept this x_new
        accept_ratio = torch.exp(
            -get_energy(x_new,t-step_size)
            + get_energy(x, t-step_size)
            + (torch.norm(x_new - x + inner_step_size * get_grad(x, t-step_size), p=2, dim=-1) ** 2)/ (4 * inner_step_size)
            -  (torch.norm(x - x_new + inner_step_size * get_grad(x_new, t-step_size), p=2, dim=-1) ** 2) ** 2 / (4 * inner_step_size)
        )
        accept = torch.rand(x.size(0)).to('cuda') < accept_ratio
        # print(accept.to(torch.int).sum() / x.shape[0])
        # x = torch.where(accept, x_new, x)
        x[accept] = x_new[accept]
    return x

def langevin_correction_alg1(x, t, step_size=1, beta=torch.tensor(0.01)):
    # record current x
    current_x = x

    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')

    # one step ddim as initial state for langevin mcmc
    # x = reverse_sde_step(x, t, step_size, beta)
    x = reverse_ddim_step(x, t, step_size, beta)
    # x = reverse_ode_step(x, t, step_size, beta)

    # avoid final redundant mcmc step
    # if t - step_size < 1e-8:
    #     return x

    # set 10 step langevin iteration
    for _ in range(20):
        # Calculate Score Components
        # weight = 4 * beta * h  # 0.0011 -> 2e-05
        weight = 2 * (1 - torch.exp(-2 * h))   # 0.0011 -> 2e-05        # weight = torch.tensor(1).to('cuda')
        # weight = 2 * (1 - torch.exp(-2 * h)) * beta   # 0.0011 -> 2e-05        # weight = torch.tensor(1).to('cuda')
        # weight = 2 * (1 - torch.exp(-2 * h)) / h ** 0.5  # 0.0011 -> 2e-05        # weight = torch.tensor(1).to('cuda')

        score = grad_log_p(x, t - step_size, beta=beta)
        term_1 = -(-2 * current_x * torch.exp(-h)) / weight
        term_2 = -(2 * x * torch.exp(-2 * h)) / weight
        # term_1 = -(-2 * current_x * torch.exp(-0.5 * beta * h)) / weight
        # term_2 = -(2 * x * torch.exp(-beta * h)) / weight

        # Calculate Score
        grad = score + term_1 + term_2
        # grad = score
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        # inner_step_size = torch.tensor(weight/100 * torch.tensor(d).to('cuda').sqrt())
        inner_step_size = torch.tensor(weight/100)
        # inner_step_size = torch.tensor(weight/100)
        x_mean = x + inner_step_size * grad
        x = x_mean + torch.sqrt(inner_step_size * 2) * noise
    return x

def langevin_correction_with_rejection_alg1(x, t, step_size=1, beta=torch.tensor(0.01)):
    # record current x
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    x = reverse_ddim_step(x, t, step_size, beta)

    # avoid final redundant mcmc step
    # if t - step_size < 1e-8:
    #     return x

    # Calculate Score Components
    # weight = 4 * beta * h
    # weight = torch.tensor(1).to('cuda')
    # weight = 2 * h
    weight = 2 * (1 - torch.exp(-2 * h))  # 0.0011 -> 2e-05
    # weight = 2 *  (1-torch.exp(-2 * h)) * beta # 0.0011 -> 2e-05
    # weight = 2 * (1 - torch.exp(-2 * h)) / h ** 0.5   # 0.0011 -> 2e-05

    inner_step_size = torch.tensor(weight / 10 * (h / 0.5) ** 1.3 )
    # inner_step_size = torch.tensor(weight / 10 * torch.tensor(d).to('cuda').sqrt())
    # inner_step_size = torch.tensor(weight / 10 * 0.2)
    # inner_step_size = torch.tensor(weight / 10 * (0.04 / 0.5) ** 1.3)
    # inner_step_size = torch.tensor(weight / 10 *
    #                                ((1 - (0.04 / 0.5)) / (0.5 - 0.04) * (h - 0.04) + (0.04 / 0.5) ** 1.3)
    #                                )
    # print(inner_step_size)
    # inner_step_size = torch.tensor(0.001 * (h / 0.04) ** 2)

    # langevin iteration
    for _ in range(20):
        def get_grad(x, t):
            score = -grad_log_p(x, t, beta=beta)
            term_1 = (-2 * current_x * torch.exp(-h)) / weight
            term_2 = (2 * x * torch.exp(-2 * h)) / weight
            # term_1 = (-2 * current_x * torch.exp(-0.5 * beta * h)) / weight
            # term_2 = (2 * x * torch.exp(-beta *  h)) / weight
            return score + term_1 + term_2
            # return score

        def get_energy(x, t):
            energy = -log_p(x, t, beta=beta)
            # term_2 = torch.square(current_x - x * torch.exp(-h)) / weight
            term_2 = torch.norm(current_x - x * torch.exp(-h), p=2, dim=-1) ** 2 / weight
            # term_2 = torch.square(current_x - x * torch.exp(-0.5 * beta * h)) / weight
            return energy + term_2
            # return energy

        # Calculate Score
        grad = get_grad(x, t - step_size)
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)

        x_mean = x - inner_step_size * grad
        x_new = x_mean + torch.sqrt(inner_step_size * 2) * noise

        # now decide whether to accept this x_new
        accept_ratio = torch.exp(
            -get_energy(x_new,t-step_size)
            + get_energy(x, t-step_size)
            + (torch.norm(x_new - x + inner_step_size * get_grad(x, t-step_size), p=2, dim=-1) ** 2)/ (4 * inner_step_size)
            -  (torch.norm(x - x_new + inner_step_size * get_grad(x_new, t-step_size), p=2, dim=-1) ** 2) ** 2 / (4 * inner_step_size)
        )
        accept = torch.rand(x.size(0)).to('cuda') < accept_ratio
        # x = torch.where(accept, x_new, x)
        # print(accept.to(torch.int).sum() / x.shape[0])
        x[accept] = x_new[accept]

    return x
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
    print(f'{marginal_acc:.6f}')
    return marginal_acc

for step_size in [40, 100, 200, 500]:
# for step_size in [500]:
    # for step_size in [100]:
    x = torch.randn([50000, d]).to('cuda')
    pbar = tqdm(total=1000)
    for t in reversed(range(1, 1001)):
        if t % step_size != 0: continue
        # x = reverse_ode_step(x, t, step_size=step_size)
        # x = reverse_sde_step(x, t, step_size=step_size)
        # x = reverse_ddim_step(x, t, step_size=step_size)
        # x = reverse_ddpm_step(x, t, step_size=step_size)
        # x = langevin_correction_explicit(x, t, step_size=step_size)
        # x = langevin_correction_alg1(x, t, step_size=step_size)
        x = langevin_correction_with_rejection_alg1(x, t, step_size=step_size)

        # x = langevin_correction(x, t, step_size=step_size)
        # x = mala(x, t, step_size=step_size)
        pbar.update(step_size)
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

    def sample_gaussian_mixture(pis=pis, d=d, mus=mus, sigmas=sigmas, num_samples=50000):
        # Determine the component for each sample
        categories = torch.distributions.Categorical(pis)
        components = categories.sample((num_samples,))

        # Prepare output tensor
        samples = torch.zeros((num_samples, d)).to('cuda')

        # Sample from each Gaussian component
        for i in range(len(pis)):
            # Number of samples from this component
            num_component_samples = (components == i).sum().item()

            # Mean and covariance of the component
            mean = mus[i]
            covariance = sigmas[i]

            # Multivariate normal distribution
            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)

            # Sampling
            samples[components == i] = mvn.sample((num_component_samples,))

        return samples


    # Generate the samples
    y = sample_gaussian_mixture(pis, d, mus, sigmas, 50000)
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
