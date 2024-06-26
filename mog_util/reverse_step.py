import torch


gaussian_mode = [0, -3]
gaussian_sigma = [1, 1]
gaussian_prob = [0.5, 0.5]

# gaussian_mode = [-1, 0, 1]
# gaussian_sigma = [0.1, 0.1, 0.1]
# gaussian_prob = torch.ones(len(gaussian_sigma)) / len(gaussian_sigma)

# gaussian_mode = [-1, -0.5, 0, 0.5, 1]
# gaussian_sigma = (0.1, ) * len(gaussian_mode)
# gaussian_prob = torch.ones(len(gaussian_sigma)) / len(gaussian_sigma)

# y = torch.concat([torch.randn(25000) * 0.3 + 1, torch.randn(25000) * 0.3 - 1]).to('cuda')
simulated_ground_truth = []
for sigma, mode, prob in zip(gaussian_sigma, gaussian_mode, gaussian_prob):
    simulated_ground_truth.append((torch.randn(int(50000 * prob)) * sigma + mode).to('cuda'))
y = torch.concat(simulated_ground_truth)

def grad_log_p(x_t, t,
               mus=torch.tensor(gaussian_mode).to('cuda'),
               # mu1=torch.tensor(1).to('cuda'),
               # mu2=torch.tensor(-1).to('cuda'),
               sigmas=torch.tensor(gaussian_sigma).to('cuda'),
               probs = torch.tensor(gaussian_prob).to('cuda'),
               beta=torch.tensor(0.01).to('cuda')):
    mus_t = mus * torch.exp(-0.5 * beta * t)
    # mu1_t = mu1 * torch.exp(-0.5 * beta * t)
    # mu2_t = mu2 * torch.exp(-0.5 * beta * t)
    sigmas_t = sigmas**2 * torch.exp(-beta * t) + 1 - torch.exp(-beta * t)

    # 计算高斯成分的概率密度
    ps = torch.exp(-0.5 * (x_t[:, None] - mus_t)**2 / sigmas_t) / torch.sqrt(2 * torch.pi * sigmas_t) * probs
    # p1 = torch.exp(-0.5 * (x_t - mu1_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)
    # p2 = torch.exp(-0.5 * (x_t - mu2_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)

    # 计算梯度对数似然
    # grad_log_p = (-(x_t - mu1_t) * p1 - (x_t - mu2_t) * p2) / (p1 + p2 + 1e-12)
    grad_log_p = -(torch.sum((x_t[:, None] - mus_t) * ps /sigmas_t, -1))  / (torch.sum(ps, -1) + 1e-12)
    # grad_log_p /= sigma_t
    return grad_log_p

def log_p(x_t, t,
          mus=torch.tensor(gaussian_mode).to('cuda'),
          # mu1=torch.tensor(1).to('cuda'),
          # mu2=torch.tensor(-1).to('cuda'),
          sigmas=torch.tensor(gaussian_sigma).to('cuda'),
          probs = torch.tensor(gaussian_prob).to('cuda'),
          beta=torch.tensor(0.01).to('cuda')):
    mus_t = mus * torch.exp(-0.5 * beta * t)
    # mu1_t = mu1 * torch.exp(-0.5 * beta * t)
    # mu2_t = mu2 * torch.exp(-0.5 * beta * t)
    sigmas_t = sigmas**2 * torch.exp(-beta * t) + 1 - torch.exp(-beta * t)

    # 计算两个高斯成分的概率密度
    ps = torch.exp(-0.5 * (x_t[:, None] - mus_t)**2 / sigmas_t) / torch.sqrt(2 * torch.pi * sigmas_t) * probs
    # p1 = torch.exp(-0.5 * (x_t - mu1_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)
    # p2 = torch.exp(-0.5 * (x_t - mu2_t)**2 / sigma_t) / torch.sqrt(2 * torch.pi * sigma_t)
    # return torch.log(0.5 * p1 + 0.5 * p2)
    return torch.log(torch.sum(ps, -1))

def reverse_sde_step(x, t, step_size=1, beta=torch.tensor(0.01).to('cuda')):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size + torch.sqrt(beta * step_size) * torch.randn_like(x)

def reverse_ode_step(x, t, step_size=1, beta=torch.tensor(0.01)):
    return x + (0.5 * beta * x + beta * grad_log_p(x, t)) * step_size

def reverse_ddim_step(x, t, step_size=1, beta=torch.tensor(0.01), eta=0):
    # alpha = 1 - beta
    # alpha_cumprod = alpha ** t
    # alpha_cumrpod_prev = alpha ** (t - step_size)
    #
    # predicted_noise = -grad_log_p(x, t) * torch.sqrt(1 -alpha_cumprod)
    # x0_t = (x - (predicted_noise * torch.sqrt((1 - alpha_cumprod)))) / torch.sqrt(alpha_cumprod)
    # # c1 = 0 * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
    # c1 = 0
    # c2 = torch.sqrt((1 - alpha_cumrpod_prev) - c1 ** 2)
    # x = torch.sqrt(alpha_cumrpod_prev) * x0_t + c2 * predicted_noise
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

def reverse_ddpm_step(x, t, step_size=1, beta=torch.tensor(0.01), **kwargs):
    x = reverse_ddim_step(x, t, step_size, beta, eta=1)
    return x

def langevin_correction(x, t, step_size=1, beta=torch.tensor(0.01)):
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    x = reverse_ddim_step(x, t, step_size, beta)

    # set 10 step langevin iteration
    for _ in range(100):
        # Calculate Score Components
        score = grad_log_p(x, t - step_size, beta=beta)

        # Calculate Score
        grad = score
        # grad = score
        noise = torch.randn_like(x).to('cuda')

        # update x (codes from langevin corrector for score sde)
        inner_step_size = torch.tensor(0.01)
        x_mean = x + inner_step_size * grad
        x = x_mean + torch.sqrt(inner_step_size * 2) * noise
    return x

def mala(x, t, step_size=1, beta=torch.tensor(0.01)):
    # one step ddim as initial state for langevin mcmc
    x = reverse_ddim_step(x, t, step_size, beta)

    # langevin iteration
    for _ in range(100):
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
        inner_step_size = torch.tensor(0.01)
        x_mean = x - inner_step_size * grad
        x_new = x_mean + torch.sqrt(inner_step_size * 2) * noise

        # now decide whether to accept this x_new
        accept_ratio = torch.exp(-get_energy(x_new,t-step_size) + get_energy(x, t-step_size) + 0.5 * ((x_new - x + inner_step_size * get_grad(x, t-step_size)) ** 2) / (2 * inner_step_size) - 0.5 * ((x - x_new + inner_step_size * get_grad(x_new, t-step_size)) ** 2) / (2 * inner_step_size))
        accept = torch.rand_like(x).to('cuda') < accept_ratio
        # print(accept.to(torch.int).sum() / x.shape[0])
        x = torch.where(accept, x_new, x)
    return x

def langevin_correction_alg1(x, t, step_size=1, beta=torch.tensor(0.01), mcmc_steps=20,
                                            mcmc_step_size_scale=1):
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
    for _ in range(mcmc_steps):
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
        inner_step_size = torch.tensor(weight/100) * mcmc_step_size_scale
        # inner_step_size = torch.tensor(weight/10)
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
    for _ in range(20):
        # Calculate Score Components
        score = grad_log_p(x, t - step_size, beta=beta)
        noise = torch.randn_like(x).to('cuda')

        # update x
        inner_step_size = torch.tensor(5e-6)
        a = 1/ ( beta * (torch.exp(2 * h) - 1))
        term_1 = score * (torch.exp(a * inner_step_size)/a - 1/a)
        term_2 = torch.exp(h) * (torch.exp(a * inner_step_size) - 1) * current_x
        term_3 = noise * torch.sqrt((torch.exp(2 * a * inner_step_size) - 1) / a)
        x =  (x + term_1 + term_2 + term_3) / torch.exp(a * inner_step_size)
    return x

def langevin_correction_with_rejection_alg1(x, t, step_size=1, beta=torch.tensor(0.01), mcmc_steps=20,
                                            mcmc_step_size_scale=1):
    # record current x
    current_x = x
    # scale step size to 0 ~ 1
    h = torch.tensor(step_size / 1000).to('cuda')
    # one step ddim as initial state for langevin mcmc
    x = reverse_ddim_step(x, t, step_size, beta)
    # x = reverse_ddpm_step(x, t, step_size, beta)

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

    inner_step_size = torch.tensor(weight / 10 * (h / 0.5) ** 1.3 * mcmc_step_size_scale)
    # inner_step_size = torch.tensor(weight /2)
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
            term_2 = torch.square(current_x - x * torch.exp(-h)) / weight
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
        accept_ratio = torch.exp(-get_energy(x_new,t-step_size) + get_energy(x, t-step_size) + 0.5 * ((x_new - x + inner_step_size * get_grad(x, t-step_size)) ** 2) / (2 * inner_step_size) - 0.5 * ((x - x_new + inner_step_size * get_grad(x_new, t-step_size)) ** 2) / (2 * inner_step_size))
        accept = torch.rand_like(x).to('cuda') < accept_ratio
        # print(accept.to(torch.int).sum() / x.shape[0])
        x = torch.where(accept, x_new, x)

    return x



reverse_step_dict = {'ddpm': reverse_ddpm_step,
                     'alg1': langevin_correction_alg1,
                     'alg1mala': langevin_correction_with_rejection_alg1}