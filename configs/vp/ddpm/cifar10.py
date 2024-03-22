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

# Lint as: python3
"""Config file for reproducing the results of DDPM on cifar-10."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  # sampling.predictor = 'ancestral_sampling'
  sampling.predictor = 'ddim'
  config.model.num_scales = 151
  # sampling.predictor = 'euler_maruyama'
  # sampling.predictor = 'adaptive'
  sampling.corrector = 'none'

  # config.sampling.method='euler_maruyama' #, choices=['euler_maruyama','adaptive']
  config.sampling.sampling_h_init=1e-2
  config.sampling.sampling_reltol=1e-2
  config.sampling.sampling_abstol=0.0078
  config.sampling.sampling_safety=0.9
  config.sampling.sampling_exp=0.9

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
