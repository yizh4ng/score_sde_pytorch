# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from tqdm import tqdm

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device,
                                abstol=config.sampling.sampling_abstol,
                                reltol=config.sampling.sampling_reltol,
                                safety=config.sampling.sampling_safety,
                                exp=config.sampling.sampling_exp,
                                adaptive=config.sampling.predictor == "adaptive",
                            h_init=config.sampling.sampling_h_init)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, h, x_prev):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, h, x_prev=None):
    # dt = -1. / self.rsde.N
    # z = torch.randn_like(x)
    # drift, diffusion = self.rsde.sde(x, t)
    # x_mean = x + drift * dt
    # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    # return x, x_mean

    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x - drift * h
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(h) * z
    return x, x_mean

@register_predictor(name='ddim')
class DDIMPredictor(Predictor):
  """Based on https://arxiv.org/pdf/2010.02502.pdf, version with no noise, only support VP process"""

  def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3, abstol = 1e-2, reltol = 1e-2,
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vpsde_update_fn(self, x, t, h, x_prev=None):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).int().to(x.device)
    timestep_next = ((t-h) * (sde.N - 1) / sde.T).int().to(x.device) # same exact thing as  timestep - 1
    # print(timestep, timestep_next)
    alpha_t = sde.alphas_cumprod.to(x.device)[timestep][:,None,None,None]
    alpha_prev = sde.alphas_cumprod.to(x.device)[timestep_next][:,None,None,None]

    predicted_noise = -self.score_fn(x, t) * torch.sqrt(1 -alpha_t)
    x0_t = (x - (predicted_noise * torch.sqrt((1 - alpha_t)))) / torch.sqrt(alpha_t)
    # c1 = 0 * torch.sqrt((1 - alpha_t / alpha_prev) * (1 - alpha_prev) / (1 - alpha_t))
    c1 = 0
    c2 = torch.sqrt((1 - alpha_prev) - c1 ** 2)
    x = torch.sqrt(alpha_prev) * x0_t + c2 * predicted_noise

    # sde = self.sde
    # timestep = (t * (sde.N - 1) / sde.T).int().to(x.device)
    # timestep_next = ((t-h) * (sde.N - 1) / sde.T).int().to(x.device) # same exact thing as  timestep - 1
    # alpha = sde.alphas_cumprod.to(x.device)[timestep][:,None,None,None]
    # alpha_next = sde.alphas_cumprod.to(x.device)[timestep_next][:,None,None,None]
    # # score = -self.score_fn(x, t) * torch.sqrt(1-alpha).to(x.device) # From Yang score-function to Ho "score-function"
    # score = -self.score_fn(x, t)
    # x = torch.sqrt(alpha_next) * (x - (torch.sqrt(1. - alpha) * score)) / torch.sqrt(alpha) + torch.sqrt(1-alpha_next) * score

    # sde = self.sde
    # timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    # timestep_next = ((t-h) * (sde.N - 1) / sde.T).astype(jnp.int32) # same exact thing as  timestep - 1
    # alpha = sde.alphas_cumprod[timestep]
    # alpha_next = sde.alphas_cumprod[timestep_next]
    # score = -batch_mul(self.score_fn(x, t), jnp.sqrt(1-alpha)) # From Yang score-function to Ho "score-function"
    # x = batch_mul(jnp.sqrt(alpha_next),batch_mul(x - batch_mul(jnp.sqrt(1. - alpha), score), 1. / jnp.sqrt(alpha))) + batch_mul(jnp.sqrt(1-alpha_next), score)
    return x, x

  def update_fn(self, x, t, h, x_prev=None):
    return self.vpsde_update_fn(x, t, h, x_prev)

# EM or Improved-Euler (Heun's method) with adaptive step-sizes
@register_predictor(name='adaptive')
class AdaptivePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
        super().__init__(sde, score_fn)
        self.h_min = 1e-10  # min step-size
        self.t = sde.T  # starting t
        self.eps = eps  # end t
        self.abstol = abstol
        self.reltol = reltol
        self.error_use_prev = True
        self.safety = safety
        self.n = shape[1] * shape[2] * shape[3]  # size of each sample
        self.exp = exp

        # "L2_scaled":
        def norm_fn(x):
            return torch.sqrt(torch.sum((x) ** 2, dim=(1, 2, 3), keepdim=True) / self.n)

        self.norm_fn = norm_fn

    def update_fn(self, x, t, h, x_prev):
        # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
        my_rsde = self.rsde.sde

        h_ = h[:, None, None, None]  # expand for multiplications
        t_ = t[:, None, None, None]  # expand for multiplications
        z = torch.randn_like(x)
        drift, diffusion = my_rsde(x, t)

        # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
        K1_mean = -h_ * drift
        K1 = K1_mean + diffusion[:, None, None, None] * torch.sqrt(h_) * z

        drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h)
        K2_mean = -h_ * drift_Heun
        K2 = K2_mean + diffusion_Heun[:, None, None, None] * torch.sqrt(h_) * z
        E = 1 / 2 * (K2 - K1)  # local-error between EM and Heun (SDEs) (right one)
        # E = 1/2*(K2_mean - K1_mean) # a little bit better with VE, but not that much
        # Extrapolate using the Heun's method result
        x_new = x + (1 / 2) * (K1 + K2)
        x_check = x + K1
        x_check_other = x_new

        # Calculating the error-control
        if self.error_use_prev:
            reltol_ctl = torch.maximum(torch.abs(x_prev), torch.abs(x_check)) * self.reltol
        else:
            reltol_ctl = torch.abs(x_check) * self.reltol
        err_ctl = torch.clamp(reltol_ctl, min=self.abstol)

        # Normalizing for each sample separately
        E_scaled_norm = self.norm_fn(E / err_ctl)

        # Accept or reject x_{n+1} and t_{n+1} for each sample separately
        accept = E_scaled_norm <= torch.ones_like(E_scaled_norm)
        x = torch.where(accept, x_new, x)
        x_prev = torch.where(accept, x_check, x_prev)
        t_ = torch.where(accept, t_ - h_, t_)

        # Change the step-size
        h_max = torch.clamp(t_ - self.eps,
                            min=0)  # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)
        E_pow = torch.where(h_ == 0, h_, torch.pow(E_scaled_norm,
                                                   -self.exp))  # Only applies power when not zero, otherwise, we get nans
        h_new = torch.minimum(h_max, self.safety * h_ * E_pow)

        return x, x_prev, t_.reshape((-1)), h_new.reshape((-1))


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t, h=None, x_prev=None):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='rtk')
class RTK(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False, shape=None, eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t, h, x_prev=None):
        # record current x (x_t)
        current_x = x

        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[timestep] # obtain current alpha. e.g., 0.98
        # h is the step size with a scale from 0 to 1
        _h = torch.Tensor([h]).to(x.device)[:, None, None, None]

        # one step euler as the initial state for langevin iteration
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x - drift * h
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(h) * z

        # one step ddpm as the initial state for langevin iteration
        # if isinstance(self.sde, sde_lib.VESDE):
        #     x, x_mean = self.vesde_update_fn(x, t)
        # elif isinstance(self.sde, sde_lib.VPSDE):
        #     x, x_mean = self.vpsde_update_fn(x, t)

        # avoid final redundant denoising step.
        if (t - h.squeeze())[0].cpu().numpy() < 1e-8:
            return x, x_mean

        # set 10 step langevin iteration
        for _ in range(10):
            # Calculate Score Components
            score = self.score_fn(x, t - h.squeeze())
            # score = self.score_fn(x, t)
            term_1 = -(-2 * current_x * torch.exp(-_h)) / (2 * (1 - torch.exp(-2 * _h)))
            term_2 = -(2 * x * torch.exp(-2 * _h)) / (2 * (1 - torch.exp(-2 * _h)))

            # Calculate Score
            grad = score + term_1 + term_2
            # grad = score
            noise = torch.randn_like(x)

            # Determine step size (codes from langevin corrector for score sde)
            # grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            # noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            # step_size = (0.16 * noise_norm / grad_norm) ** 2 * 2 * alpha
            step_size = torch.Tensor((4e-05, ) * x.shape[0]).to(x.device)
            # print(step_size)

            # update x (codes from langevin corrector for score sde)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean

@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, h, x_prev, sde, model, predictor, probability_flow, continuous, shape=None,
                               eps=1e-3, abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow, shape=shape, eps=eps,
                              abstol=abstol, reltol=reltol, safety=safety, exp=0.9)
  return predictor_obj.update_fn(x, t, h, x_prev)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', abstol=1e-2,
                   reltol=1e-2, safety=.9, exp=0.9, adaptive=False, h_init=1e-2):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous, shape=shape,
                                          eps=eps, abstol=abstol, reltol=reltol, safety=safety, exp=exp)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      # timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      timesteps = np.linspace(sde.T, eps, sde.N)
      h = timesteps - np.append(timesteps, 0)[
                      1:]  # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)
      timesteps = torch.from_numpy(timesteps).to(x.device)
      with tqdm(total=sde.N) as pbar:
          for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            # vec_t = torch.ones(shape[0]).to(x.device) * t
            x, x_mean = corrector_update_fn(x, vec_t, model=model)
            x, x_mean = predictor_update_fn(x, vec_t, h=h[i], x_prev=None, model=model)
            pbar.update(1)

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  # return pc_sampler
  def pc_sampler_adaptive(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
        # print(datetime.now().time())
        # Initial sample
        x = sde.prior_sampling(shape).to(device)
        h = torch.ones(shape[0]).to(x.device) * h_init  # initial step_size
        t = torch.ones(shape[0]).to(x.device) * sde.T  # initial time
        x_prev = x

        N = 0

        with tqdm(total=sde.N) as pbar:
            while (torch.abs(t - eps) > 1e-6).any():
                x, x_prev, t, h = predictor_update_fn(x, t, h, x_prev=x_prev, model=model)
                N = N + 1
                pbar.update(1)

        if denoise:
            eps_t = torch.ones(shape[0]).to(x.device) * eps
            u, std = sde.marginal_prob(x, eps_t)
            x = x + (std[:, None, None, None] ** 2).to(x.device) * model(x, eps_t).to(x.device)
        # print(datetime.now().time())
        return inverse_scaler(x), N + 1

  if adaptive:
    return pc_sampler_adaptive
  else:
    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler