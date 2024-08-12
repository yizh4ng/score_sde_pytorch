import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

from mnist_fid_util import evaluate_fid_score


class ScoreNetwork0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal

score_network = ScoreNetwork0()
device = torch.device('cuda:5')  # change this if you don't have a gpu
score_network = score_network.to(device)

score_network.load_state_dict(torch.load(f'./mnist_better.pth'))
# score_network.load_state_dict(torch.load(f'./mnist_1.pth'))


def generate_samples(score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
    from sde_lib_mnist import VPSDE
    sde =  VPSDE()
    rsde = sde.reverse(score_network, probability_flow=False)
    device = next(score_network.parameters()).device
    x_t = torch.randn((nsamples, 28 * 28), device=device)  # (nsamples, nch)

    time_pts = torch.linspace(1, 0, 1000, device=device)  # (ntime_pts,)
    beta = lambda t: 0.1 + (20 - 0.1) * t
    for i in range(len(time_pts)-1):
        # t = time_pts[i]
        # dt = time_pts[i + 1] - t
        # fxt = -0.5 * beta(t) * x_t
        # gt = beta(t) ** 0.5
        # score = score_network(x_t, t.expand(x_t.shape[0], 1)).detach()
        # drift = fxt - gt * gt * score
        # diffusion = gt
        # x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

        # reverse sde # 25.95
        if i < 400: continue
        t = time_pts[i].unsqueeze(-1)
        dt = time_pts[i + 1] - t
        t = t.expand(x_t.shape[0], 1)
        drift, diffusion = rsde.sde(x_t, t)
        # euler-maruyama step
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5


        # ddim or ddpm # 20.4 21.29
        # sde = sde
        # t = time_pts[i]
        # timestep = (t * (sde.N - 1) / sde.T).int().to(x_t.device)
        # h = 1 / len(time_pts)
        # timestep_next = ((t-h) * (sde.N - 1) / sde.T).int().to(x_t.device) # same exact thing as  timestep - 1
        # alpha_t = sde.alphas_cumprod.to(x_t.device)[timestep]
        # alpha_prev = sde.alphas_cumprod.to(x_t.device)[timestep_next]
        # t = t.expand(x_t.shape[0], 1)
        # predicted_noise = -score_network(x_t, t).detach() * torch.sqrt(1 -alpha_t)
        # x0_t = (x_t - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t)
        # c1 = 0 * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
        # c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
        # x_t = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * torch.randn_like(x_t)

        # ald 33.14
        # # if i in [100, 200, 300, 400, 500, 600, 700, 800,900, 920, 940, 960, 980, 999]:
        # if i in [100,150,200,250, 300,350,400,450,500,550,
        #          600, 650, 700, 750, 800, 850, 900, 920, 940, 960, 980, 999]:
        # # if i in range(0, 1000, 20):
        #     for j in range(100):
        #         t = time_pts[i]
        #         timestep = (t * (sde.N - 1) / sde.T).int().to(x_t.device)
        #         alpha = sde.alphas[timestep]
        #
        #         t = t.expand(x_t.shape[0], 1)
        #         grad = score_network(x_t, t).detach()
        #         noise = torch.randn_like(x_t)
        #         std = sde.marginal_prob(x_t, t)[1]
        #         # step_size = torch.tensor([0.5]).to(device)
        #         step_size = (0.2 * std) ** 2 * 2 * alpha
        #         x_mean = x_t + step_size * grad / std
        #         x_t = x_mean + noise * torch.sqrt(step_size * 2)

    ## rtk uld
    # # # timesteps = [600, 700, 800, 900, 920, 940, 960, 980, 999]
    # time_pts = torch.linspace(1, 0, 1000, device=device)  # (ntime_pts,)
    # # timesteps = [100, 200, 300, 400, 500, 600, 700, 800, 830, 860, 900, 920, 940, 960, 980, 999]
    # # timesteps = [100, 200, 300, 400, 500, 600, 700, 800,  900,  999]
    # # timesteps = [300, 600, 700, 800, 900, 920, 940, 960, 980, 999]
    # # timesteps = [100, 300, 600,  800, 850, 900, 920, 940, 960, 980, 999]
    # # timesteps = [100, 600, 700, 800, 900, 920, 940, 950, 960, 970, 980, 990, 999]
    # # timesteps = [100, 300, 600, 700, 800, 900, 920, 940, 960, 980, 999]
    # # timesteps.insert(0, 0)
    # import numpy as np
    # timesteps = np.linspace(0, 999, 10, endpoint=True).astype(np.int64)
    # weight_scale = 1
    # for index, i in enumerate(timesteps[:-1]):
    #     current_x = x_t
    #
    #     sde = sde
    #     t = time_pts[i]
    #     t_prev = time_pts[timesteps[index + 1]]
    #     h = t-t_prev
    #     timestep = (t * (sde.N - 1) / sde.T).int().to(x_t.device)
    #     timestep_next = ((t-h) * (sde.N - 1) / sde.T).int().to(x_t.device) # same exact thing as  timestep - 1
    #     alpha_t = sde.alphas_cumprod.to(x_t.device)[timestep]
    #     alpha_prev = sde.alphas_cumprod.to(x_t.device)[timestep_next]
    #     t = t.expand(x_t.shape[0], 1)
    #     predicted_noise = -score_network(x_t, t).detach() * torch.sqrt(1 -alpha_t)
    #     x0_t = (x_t - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t)
    #     c1 = 0 * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
    #     c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
    #     x_t = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise + c1 * torch.randn_like(x_t)
    #     if index == (len(timesteps[:-1]) - 1): return x_t
    #
    #     # sde = VPSDE()
    #     # rsde = sde.reverse(score_network, probability_flow=True)
    #     # t = time_pts[timesteps[index]]
    #     # t_prev = time_pts[timesteps[index + 1]]
    #     # h = t-t_prev
    #     # dt = -h
    #     # t = t.expand(x_t.shape[0], 1)
    #     # drift, diffusion = rsde.sde(x_t, t)
    #     # # euler-maruyama step
    #     # x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
    #
    #
    #     # beta = sde.discrete_betas.to(t.device)[timestep]
    #     # weight = 4 * beta * h # 0.0011 -> 2e-05
    #     # _beta = 0.5* (beta(1-t_prev) + beta(1-t))
    #     _beta = beta(1-t_prev)
    #     # _beta = 2
    #     weight = 2 * (1 - torch.exp(-2 * _beta * h))
    #     # weight = 4 * h
    #     std = sde.marginal_prob(x_t, t)[1]
    #     # ula
    #     for j in range(1):
    #             t = time_pts[i]
    #             timestep = (t * (sde.N - 1) / sde.T).int().to(x_t.device)
    #             alpha = sde.alphas[timestep]
    #
    #             t = t.expand(x_t.shape[0], 1)
    #             grad = score_network(x_t, t).detach() / std
    #             term_1 = -(-2 * current_x * torch.exp(-h * _beta)) / weight
    #             term_2 = -(2 * x_t * torch.exp(-2 * h * _beta)) / weight
    #             grad = grad + term_1 + term_2
    #
    #             noise = torch.randn_like(x_t)
    #             # step_size = torch.tensor([5e-6]).to(device)
    #             step_size = (0.2 * std) ** 2 * 2 * alpha * 0.1#* weight * 1
    #             print(step_size)
    #             # step_size = torch.sqrt((0.2 * std) ** 2 * 2 * alpha * weight/200)
    #             # step_size = weight /  2000
    #             # step_size = torch.tensor(weight/100)
    #             x_mean = x_t + step_size * grad
    #             x_t = x_mean + noise * torch.sqrt(step_size * 2)
    #     # uld
    #     # v = torch.randn_like(x_t).to(device)
    #     # for j in range(20):
    #     #     t = time_pts[i]
    #     #     timestep = (t * (sde.N - 1) / sde.T).int().to(x_t.device)
    #     #     alpha = sde.alphas[timestep]
    #     #
    #     #     std = sde.marginal_prob(x_t, t)[1]
    #     #     t = t.expand(x_t.shape[0], 1)
    #     #     grad = score_network(x_t, t).detach() / std
    #     #     term_1 = -(-2 * current_x * torch.exp(-h * _beta)) / weight
    #     #     term_2 = -(2 * x_t * torch.exp(-2 * h * _beta)) / weight
    #     #     grad = grad + term_1 + term_2
    #     #
    #     #     # inner_step_size = torch.tensor([5e-4]).to(device)
    #     #     inner_step_size = (0.2 * std) ** 2 * 2 * alpha * weight
    #     #     # inner_step_size = (0.2 * std) ** 2 * 2 * alpha
    #     #     # inner_step_size = weight / 200
    #     #     # step_size = torch.tensor(weight/100)
    #     #
    #     #     gamma =  0.1
    #     #     x_t = x_t + v * inner_step_size
    #     #     v = (v - gamma * v * inner_step_size
    #     #          + grad * inner_step_size
    #     #          + torch.sqrt(torch.tensor(2 * gamma * inner_step_size)) * torch.randn_like(v))
    return x_t

samples = generate_samples(score_network, 21).detach().reshape(-1, 28, 28)
nrows, ncols = 3, 7
plt.figure(figsize=(3 * ncols, 3 * nrows))
for i in range(samples.shape[0]):
    plt.subplot(nrows, ncols, i + 1)
    plt.imshow(1 - samples[i].detach().cpu().numpy(), cmap="Greys")
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.close()

import numpy as np
all_samples = []
for i in range(1):
    samples = generate_samples(score_network, 1000).detach().reshape(-1, 28, 28)
    samples_np = samples.cpu().numpy()
    # samples_np = samples_np - np.nanmin(samples_np, axis=(1,2))[:, None, None]
    # samples_np = samples_np / np.nanmax(samples_np, axis=(1,2))[:, None, None]
    samples_np = np.nan_to_num(samples_np, nan=0.0)
    # all_samples.append(np.clip(samples_np, 0, 1))
    all_samples.append(samples_np)
samples_np = np.concatenate(all_samples, axis=0)
# np.savez("./ula.npz", samples=samples_np)

transforms = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((299, 299)),
    # torchvision.transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),  # 归一化
])

mnist_dset = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)
# mnist_dset = torchvision.datasets.MNIST("mnist", download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_dset, batch_size=1000, shuffle=True)

# 获取一个batch的MNIST数据
mnist_images, _ = next(iter(mnist_loader))
mnist_images = mnist_images.repeat(1, 3, 1, 1)
# mnist_images = 1-mnist_images
# mnist_images = (mnist_images + 1) * 0.5
# mnist_images = (mnist_images - 0.5) * 0.2 + 0.5
mnist_images = mnist_images * 0.2 + 0.5

generated_samples = samples_np
generated_samples = torch.tensor(np.stack(generated_samples, axis=0)).to('cuda')
generated_samples = generated_samples.unsqueeze(1)
generated_samples = generated_samples.repeat(1, 3, 1, 1)
# 转换生成的样本并提取特征
# generated_samples = np.stack(
#     [np.array(transforms(Image.fromarray((sample * 255).astype(np.uint8)))) for sample in
#      generated_samples])
# generated_samples = 1 - generated_samples
# generated_samples = (generated_samples + 1) * 0.5
# generated_samples = (generated_samples - 0.5) * 0.2 + 0.5
generated_samples = generated_samples * 0.2 + 0.5
fid = evaluate_fid_score(mnist_images, generated_samples, dim=2048, batch_size=1000)
print(fid)
# from scipy.stats import wasserstein_distance
# mnist_images = mnist_images[:, 0, :, :]
# mnist_images = mnist_images.view(-1, 28*28)
# generated_samples = generated_samples[:, 0, :, :]
# generated_samples = generated_samples.view(-1, 28*28)
# distance = wasserstein_distance(mnist_images.to('cpu').numpy(), generated_samples.to('cpu').numpy())
# print(f"The Wasserstein distance between the two distributions is: {distance}")