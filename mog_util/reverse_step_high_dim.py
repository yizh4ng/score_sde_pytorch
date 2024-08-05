import torch
from torch.distributions import MultivariateNormal

from mog_util.mog_high_dim_config import pis, mus, sigmas, grad_log_p_noise, log_p_noise, d, ground_truth_num, synthetic_num

grad_noise = torch.randn(d).to('cuda') * grad_log_p_noise
p_noise = torch.randn(1).to('cuda') * log_p_noise

def grad_log_p(x_t, t, beta=torch.tensor(0.01).to('cuda')):
    n_samples = x_t.shape[0]  # Number of samples in the batch
    d = x_t.shape[1]  # Dimensionality of each sample
    k = pis.shape[0]  # Number of mixture components

    # Ensure mus and sigmas are properly expanded for batch operations
    mu_t = mus * torch.exp(-0.5 * beta * t)
    sigma_t = sigmas * torch.exp(-beta * t) + (1 - torch.exp(-beta * t)) * torch.eye(d, device=x_t.device)

    # Inverse and determinant of covariance matrix
    sigma_t_inv = torch.inverse(sigma_t)
    det_sigma_t = torch.det(sigma_t)

    # Broadcasting mu_t for each sample and mixture component
    mu_t_expanded = mu_t.unsqueeze(1).expand(k, n_samples, d)
    x_t_expanded = x_t.unsqueeze(0).expand(k, n_samples, d)
    diff = x_t_expanded - mu_t_expanded

    # Perform matrix multiplication and exponent calculation with einsum
    exp_component = torch.exp(-0.5 * torch.einsum('kni,kij,knj->kn', diff, sigma_t_inv, diff))
    normal_density = exp_component / torch.sqrt((2 * torch.pi) ** d * det_sigma_t.unsqueeze(1))

    # Weight the density by the mixture coefficients
    weighted_density = pis.unsqueeze(1) * normal_density
    p_x_t = weighted_density.sum(0)

    # Accumulate the weighted sum for the gradient calculation
    weighted_sum = torch.einsum('kn,kni->ni', weighted_density, torch.einsum('kij,knj->kni', sigma_t_inv, diff))

    # Compute the gradient of the log probability
    grad_log_p = -weighted_sum / (p_x_t.unsqueeze(-1) + 1e-15)
    if grad_log_p_noise != 0:
        # noise_std = grad_log_p_noise * torch.abs(grad_log_p)
        # grad_log_p = grad_log_p + torch.ones_like(grad_log_p) * noise_std
        # grad_log_p = grad_log_p + torch.randn_like(grad_log_p) * noise_std
        grad_log_p = grad_log_p + grad_log_p_noise
    return grad_log_p

def log_p(x_t, t, beta=torch.tensor(0.01).to('cuda')):
    n_samples = x_t.shape[0]  # Number of samples in the batch
    d = x_t.shape[1]  # Dimensionality of each sample
    k = pis.shape[0]  # Number of mixture components

    # Ensure mus and sigmas are properly expanded for batch operations
    mu_t = mus * torch.exp(-0.5 * beta * t)
    sigma_t = sigmas * torch.exp(-beta * t) + (1 - torch.exp(-beta * t)) * torch.eye(d, device=x_t.device)

    # Inverse and determinant of covariance matrix
    sigma_t_inv = torch.inverse(sigma_t)
    det_sigma_t = torch.det(sigma_t)

    # Broadcasting mu_t for each sample and mixture component
    mu_t_expanded = mu_t.unsqueeze(1).expand(k, n_samples, d)
    x_t_expanded = x_t.unsqueeze(0).expand(k, n_samples, d)
    diff = x_t_expanded - mu_t_expanded

    # Perform matrix multiplication and exponent calculation with einsum
    exp_component = torch.exp(-0.5 * torch.einsum('kni,kij,knj->kn', diff, sigma_t_inv, diff))
    normal_density = exp_component / torch.sqrt((2 * torch.pi) ** d * det_sigma_t.unsqueeze(1))

    # Weight the density by the mixture coefficients
    weighted_density = pis.unsqueeze(1) * normal_density
    p_x_t = weighted_density.sum(0)
    # return torch.log(p_x_t)
    log_p_x_t = torch.log(p_x_t)

    if log_p_noise != 0:
        noise_std = log_p_noise * torch.abs(log_p_x_t)
        # log_p_x_t = log_p_x_t + torch.randn_like(log_p_x_t) * noise_std
        # log_p_x_t = log_p_x_t + noise[None, 0] * noise_std
        # log_p_x_t = log_p_x_t + torch.ones_like(log_p_x_t) * noise_std
        log_p_x_t = log_p_x_t + log_p_noise
    return log_p_x_t

def reverse_sde_step(x, t, step_size=1, beta=torch.tensor(0.01).to('cuda'), **kwargs):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size + torch.sqrt(beta * step_size) * torch.randn_like(x)

def reverse_ode_step(x, t, step_size=1, beta=torch.tensor(0.01), **kwargs):
    return x + (0.5 * beta * x + 0.5 * beta * grad_log_p(x, t)) * step_size
def reverse_ddim_step(x, t, step_size=1, beta=torch.tensor(0.01), eta=0, **kwargs):
    alpha = 1 - beta
    alpha_cumprod = alpha ** t
    alpha_cumrpod_prev = alpha ** (t - step_size)

    predicted_noise = -grad_log_p(x, t) * torch.sqrt(1 -alpha_cumprod)

    alpha_t = alpha_cumprod
    alpha_prev = alpha_cumrpod_prev
    x0_t = (x - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t)
    c1 = eta * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
    c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
    x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * torch.randn_like(x)
    return x

    # epsilon_t = torch.randn_like(x)
    # sigma_t = eta * torch.sqrt((1 - alpha_cumrpod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumrpod_prev))
    #
    # x = (
    #         torch.sqrt(alpha_cumrpod_prev / alpha_cumprod) * x +
    #         (torch.sqrt(1 - alpha_cumrpod_prev - sigma_t ** 2) - torch.sqrt(
    #             (alpha_cumrpod_prev * (1 - alpha_cumprod)) / alpha_cumprod)) * predicted_noise +
    #         sigma_t * epsilon_t
    # )
    # return x

def reverse_ddpm_drift_step(x, t, step_size=1, beta=torch.tensor(0.01), eta=1):
    alpha = 1 - beta
    alpha_cumprod = alpha ** t
    alpha_cumrpod_prev = alpha ** (t - step_size)

    predicted_noise = -grad_log_p(x, t) * torch.sqrt(1 -alpha_cumprod)

    sigma_t = eta * torch.sqrt((1 - alpha_cumrpod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumrpod_prev))

    x = (
            torch.sqrt(alpha_cumrpod_prev / alpha_cumprod) * x +
            (torch.sqrt(1 - alpha_cumrpod_prev - sigma_t ** 2) - torch.sqrt(
                (alpha_cumrpod_prev * (1 - alpha_cumprod)) / alpha_cumprod)) * predicted_noise
    )
    return x
def reverse_ddpm_step(x, t, step_size=1, beta=torch.tensor(0.01), **kwargs):
    x = reverse_ddim_step(x, t, step_size, beta, eta=1)
    return x

def langevin_correction(x, t, step_size=1, beta=torch.tensor(0.01),
                        mcmc_steps=20, mcmc_step_size_scale=1, **kwargs):
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    x = reverse_ddpm_step(x, t, step_size, beta)

    # set 10 step langevin iteration
    for _ in range(mcmc_steps):
        # Calculate Score Components
        score = grad_log_p(x, t - step_size, beta=beta)

        # Calculate Score
        grad = score
        # grad = score
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        inner_step_size = torch.tensor(0.5) * mcmc_step_size_scale
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

def langevin_correction_alg1(x, t, step_size=1, beta=torch.tensor(0.01), mcmc_steps=20, mcmc_step_size_scale=1,
                             init=None, weight_scale=1):
    # record current x
    current_x = x

    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')

    # one step ddim as initial state for langevin mcmc
    # x = reverse_sde_step(x, t, step_size, beta)
    if init is not None:
        if init == 'ddim':
            x = reverse_ddim_step(x, t, step_size, beta)
        elif init == 'ddpm':
            x = reverse_ddpm_step(x, t, step_size, beta)
        elif init == 'ddpm_drift':
            x = reverse_ddpm_drift_step(x, t, step_size, beta)
        else:
            raise NotImplementedError
        # x = reverse_ddpm_step(x, t, step_size, beta)
    # x = reverse_ode_step(x, t, step_size, beta)

    # avoid final redundant mcmc step
    # if t - step_size < 1e-8:
    #     return x

    # set 10 step langevin iteration
    for _ in range(mcmc_steps):
        # Calculate Score Components
        # weight = 4 * h  # 0.0011 -> 2e-05
        weight = 2 * (1 - torch.exp(-2 * h)) * weight_scale  # 0.0011 -> 2e-05        # weight = torch.tensor(1).to('cuda')
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

        # inner_step_size = torch.tensor(weight/100) * mcmc_step_size_scale * (1000/step_size) / 25
        # inner_step_size = torch.tensor(weight / 10 * (h / 0.5) ** 1) * mcmc_step_size_scale * (1000 / step_size) / 25
        # inner_step_size = torch.tensor(weight/100) * step_size / 100 * mcmc_step_size_scale
        inner_step_size = torch.tensor(weight/100) * mcmc_step_size_scale

        # inner_step_size = torch.tensor(weight/100)
        x_mean = x + inner_step_size * grad
        x = x_mean + torch.sqrt(inner_step_size * 2) * noise
    return x

def uld(x, t, step_size=1, beta=torch.tensor(0.01), mcmc_steps=20, mcmc_step_size_scale=1,
                             init=None, weight_scale=1):
    # record current x
    current_x = x

    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')

    # one step ddim as initial state for langevin mcmc
    if init is not None:
        if init == 'ddim':
            x = reverse_ddim_step(x, t, step_size, beta)
        elif init == 'ddpm':
            x = reverse_ddpm_step(x, t, step_size, beta)
        elif init == 'ddpm_drift':
            x = reverse_ddpm_drift_step(x, t, step_size, beta)
        elif init == 'gaussian':
            x = x + torch.randn_like(x).to('cuda') * torch.sqrt(torch.exp(2 * h) - 1)
        else:
            raise NotImplementedError


    # set 10 step langevin iteration
    v = torch.randn_like(x).to('cuda')

    gamma = 50
    # tau = torch.tensor(mcmc_step_size_scale, dtype=torch.float)
    #
    # var_x = 2 / gamma * (tau - 2 / gamma * (1 - torch.exp(-gamma * tau))) + 1 / (2 * gamma) * (
    #             1 - torch.exp(-2 * gamma * tau))
    # var_v = 1 - torch.exp(-2 * gamma * tau)
    # cov_xv = 1 / gamma * (1 - 2 * torch.exp(-gamma * tau) + torch.exp(-2 * gamma * tau))
    #
    # # 构建完整的协方差矩阵
    # Sigma_xx = var_x * torch.eye(d)
    # Sigma_vv = var_v * torch.eye(d)
    # Sigma_xv = cov_xv * torch.eye(d)
    #
    # # 构建 2d x 2d 协方差矩阵
    # top_row = torch.cat([Sigma_xx, Sigma_xv], dim=1)
    # bottom_row = torch.cat([Sigma_xv, Sigma_vv], dim=1)
    # cov_matrix = torch.cat([top_row, bottom_row], dim=0)
    #
    # # 创建多元正态分布
    # mvn = MultivariateNormal(torch.zeros(2 * d), cov_matrix)

    for _ in range(mcmc_steps):
        # Calculate Score Components
        weight = 2 * (1 - torch.exp(-2 * h))   # 0.0011 -> 2e-05        # weight = torch.tensor(1).to('cuda')

        score = grad_log_p(x, t - step_size, beta=beta)
        term_1 = -(-2 * current_x * torch.exp(-h)) / weight
        term_2 = -(2 * x * torch.exp(-2 * h)) / weight

        # Calculate Score
        grad = score + term_1 + term_2

        # 采样
        # noise = mvn.sample((x.size(0),))  # 生成样本
        # noise_x = noise[:, :d].to('cuda')
        # noise_v = noise[:, d:].to('cuda')
        # x = (x
        #      + gamma ** -1 * (1 - torch.exp(-gamma * tau)) * v
        #      + gamma ** -1 * (1 - torch.exp(-gamma * tau)) * grad
        #      + noise_x
        #      )
        # v = (torch.exp(-gamma * tau) * v
        #      + gamma ** -1 * (1 - torch.exp(-gamma * tau)) * grad
        #      + noise_v
        #      )

        inner_step_size = mcmc_step_size_scale
        gamma = 30
        v = (v - gamma * v * inner_step_size
             + grad * inner_step_size
             + torch.sqrt(torch.tensor(2 * gamma * inner_step_size)) * torch.randn_like(v))
        x = x + v * inner_step_size

    return x
def langevin_correction_with_rejection_alg1(x, t, step_size=1, beta=torch.tensor(0.01),
                                            mcmc_steps=20, mcmc_step_size_scale=1, init=None, weight_scale=1):
    # record current x
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    if init is not None:
        if init == 'ddim':
            x = reverse_ddim_step(x, t, step_size, beta)
        elif init == 'ddpm':
            x = reverse_ddpm_step(x, t, step_size, beta)
        elif init == 'ddpm_drift':
            x = reverse_ddpm_drift_step(x, t, step_size, beta)
        else:
            raise NotImplementedError

            # avoid final redundant mcmc step
    # if t - step_size < 1e-8:
    #     return x

    # Calculate Score Components
    # weight = 4 * beta * h
    # weight = torch.tensor(1).to('cuda')
    # weight = 2 * h
    weight = 2 * (1 - torch.exp(-2 * h)) * weight_scale# 0.0011 -> 2e-05
    # weight = 2 *  (1-torch.exp(-2 * h)) * beta # 0.0011 -> 2e-05
    # weight = 2 * (1 - torch.exp(-2 * h)) / h ** 0.5   # 0.0011 -> 2e-05

    inner_step_size = torch.tensor(weight / 10 * (h / 0.5) ** 1 ) * mcmc_step_size_scale * (1000/step_size) / 25
    # inner_step_size = torch.tensor(weight / 100) * mcmc_step_size_scale
    # inner_step_size = torch.tensor(weight / 100) *  step_size / 100 * mcmc_step_size_scale
    # inner_step_size = torch.tensor(weight / 10 * torch.tensor(d).to('cuda').sqrt())
    # inner_step_size = torch.tensor(weight / 10 * 0.2)
    # inner_step_size = torch.tensor(weight / 10 * (0.04 / 0.5) ** 1.3)
    # inner_step_size = torch.tensor(weight / 10 *
    #                                ((1 - (0.04 / 0.5)) / (0.5 - 0.04) * (h - 0.04) + (0.04 / 0.5) ** 1.3)
    #                                )
    # print(inner_step_size)
    # inner_step_size = torch.tensor(0.001 * (h / 0.04) ** 2)

    # langevin iteration
    for _ in range(mcmc_steps):
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
            - get_energy(x_new,t-step_size)
            + get_energy(x, t-step_size)
            + (torch.norm(x_new - x - inner_step_size * get_grad(x, t-step_size), p=2, dim=-1) ** 2)/ (4 * inner_step_size)
            -  (torch.norm(x - x_new - inner_step_size * get_grad(x_new, t-step_size), p=2, dim=-1) ** 2) ** 2 / (4 * inner_step_size)
        )
        accept = torch.rand(x.size(0)).to('cuda') < accept_ratio
        # x = torch.where(accept, x_new, x)
        # print(accept.to(torch.int).sum() / x.shape[0])
        x[accept] = x_new[accept]

    return x

def langevin_correction_with_rejection_alg1_estimated(x, t, step_size=1, beta=torch.tensor(0.01),
                                            mcmc_steps=20, mcmc_step_size_scale=1, init=None, weight_scale=1):
    # record current x
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    if init is not None:
        if init == 'ddim':
            x = reverse_ddim_step(x, t, step_size, beta)
        elif init == 'ddpm':
            x = reverse_ddpm_step(x, t, step_size, beta)
        elif init == 'ddpm_drift':
            x = reverse_ddpm_drift_step(x, t, step_size, beta)
        else:
            raise NotImplementedError

            # avoid final redundant mcmc step
    # if t - step_size < 1e-8:
    #     return x

    # Calculate Score Components
    weight = 2 * (1 - torch.exp(-2 * h)) * 1# 0.0011 -> 2e-05

    inner_step_size = torch.tensor(weight / 10 * (h / 0.5) ** 1 ) * mcmc_step_size_scale * (1000/step_size) / 25

    # langevin iteration
    for _ in range(mcmc_steps):
        def get_grad(x, t):
            score = -grad_log_p(x, t, beta=beta)
            term_1 = (-2 * current_x * torch.exp(-h)) / weight
            term_2 = (2 * x * torch.exp(-2 * h)) / weight
            return score + term_1 + term_2

        def explicit_quadratic(x):
            term_2 = torch.norm(current_x - x * torch.exp(-h), p=2, dim=-1) ** 2 / weight
            return term_2

        def esimated_energy_difference_1(x_new, x):
            # estimate log_p(x_new) - log_p(x)
            eps = 0.2
            def h(i, delta_t):
                if i == 1:
                    return (torch.einsum( 'ij,ij->i', (grad_log_p((x_new - x) * (delta_t + eps) + x, t - step_size) , (x_new - x)))
                            / eps)
                else:
                    return (h(i - 1, eps) - h(i - 1, 0)) / eps

            estimated_energy_difference = 0

            for i in range(1, 4):
                import math
                estimated_energy_difference += h(i, 0) / math.factorial(i)
                # print(torch.mean(estimated_energy_difference))

            return estimated_energy_difference

        def esimated_energy_difference(x_new, x):
            # estimate log_p(x_new) - log_p(x)
            eps = 0.005
            # eps = 0.00001
            mid_point = 0.0
            def h(i, delta_t):
                if i == 1:
                    return torch.einsum('ij,ij->i', (grad_log_p((x_new - x) * mid_point + x, t - step_size), (x_new - x)))
                elif i == 2:
                    f = lambda _delta_t, _eps: torch.einsum('ij,ij->i', (grad_log_p((x_new - x) * (mid_point + _delta_t + _eps) + x, t - step_size), (x_new - x)))
                    return (f(delta_t, eps) - f(delta_t, -eps))/ (2 * eps)
                elif i > 2:
                    return (h(i - 1, eps) - h(i - 1, -eps)) / (2 * eps)

                    # return (h(i - 1, eps) - h(i - 1, 0)) / eps

            estimated_energy_difference = 0

            for i in range(1, 3):
                import math
                estimated_energy_difference += h(i, 0) / math.factorial(i)
            return estimated_energy_difference

        # Calculate Score
        grad = get_grad(x, t - step_size)
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)

        x_mean = x - inner_step_size * grad
        x_new = x_mean + torch.sqrt(inner_step_size * 2) * noise

        # now decide whether to accept this x_new
        accept_ratio = torch.exp(
            # -get_energy(x_new,t-step_size)
            # + get_energy(x, t-step_size)
            + esimated_energy_difference(x_new, x)
            - explicit_quadratic(x_new)
            + explicit_quadratic(x)
            + (torch.norm(x_new - x - inner_step_size * get_grad(x, t-step_size), p=2, dim=-1) ** 2)/ (4 * inner_step_size)
            -  (torch.norm(x - x_new - inner_step_size * get_grad(x_new, t-step_size), p=2, dim=-1) ** 2) ** 2 / (4 * inner_step_size)
        )
        accept = torch.rand(x.size(0)).to('cuda') < accept_ratio
        # x = torch.where(accept, x_new, x)
        # print(accept.to(torch.int).sum() / x.shape[0])
        x[accept] = x_new[accept]

    return x

reverse_step_dict = {'DDPM': reverse_ddpm_step,
                     'DDIM': reverse_ddim_step,
                     'SDE':reverse_sde_step,
                     'ODE':reverse_ode_step,
                     'Langevin': langevin_correction_alg1,
                     'ALD': langevin_correction,
                     'ULD': uld,
                     'MALA': langevin_correction_with_rejection_alg1,
                     'MALA_ES': langevin_correction_with_rejection_alg1_estimated,
                     }


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
y = sample_gaussian_mixture(pis, d, mus, sigmas, ground_truth_num )
