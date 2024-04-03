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

"""Training and evaluation"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 4, 5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4, 5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging

import tensorflow as tf

FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file(
#   "config", 'configs/vp/ddpm/cifar10.py', "Training configuration.", lock_config=True)
# config_flags.DEFINE_config_file(
#   "config", 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py', "Training configuration.", lock_config=True)
config_flags.DEFINE_config_file(
  "config", 'configs/ddpmpp/cifar10_continuous_vp_c8.py', "Training configuration.", lock_config=True)
# flags.DEFINE_string("workdir", 'ddpmpp_vp_continuous_ddim', "Work directory.")
# flags.DEFINE_string("workdir", 'ddpm_vp_adaptive', "Work directory.")
# flags.DEFINE_string("workdir", 'test', "Work directory.")
flags.DEFINE_string("workdir", 'ddpmpp_vp_alg1_ddim_re', "Work directory.")
# flags.DEFINE_string("workdir", 'ddpmpp_vp_alg1_ddim', "Work directory.")
flags.DEFINE_enum("mode", 'eval', ["train", "eval", "fid_stats"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("fid_folder", "assets/stats",
                    "The folder name for storing FID statistics")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)

  elif FLAGS.mode == "fid_stats":
    # Calculate the FID statistics
    run_lib.fid_stats(FLAGS.config, FLAGS.fid_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
