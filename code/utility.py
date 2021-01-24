# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Utility Module
# Utility functions to run model-agnostic meta-learning supervised regression experiments.

import argparse
import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({"font.size": 12})


def control_randomness(seed):
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def transform_array_to_tensor(array, torch_device, torch_dtype=None):
  return torch.from_numpy(array).to(torch_device, dtype=torch_dtype)


def transform_tensor_to_array(tensor, numpy_dtype=None):
  if numpy_dtype:
    return tensor.detach().numpy().astype(numpy_dtype)
  else:
    return tensor.detach().numpy()


def format_algorithm_name_for_plot(algorithm_name):
  if algorithm_name == "baseline":
    return "Baseline"
  elif algorithm_name == "maml":
    return "MAML"


def create_and_save_test_result_plots(directory_path, algorithm_name, loss_array,
                                      sine_wave_approximation_prediction_array, sine_wave_input_array,
                                      sine_wave_output_array, sine_wave_approximation_input_array,
                                      sine_wave_approximation_output_array):
  pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)

  plt.plot(loss_array)
  plt.title("{} training loss during testing".format(format_algorithm_name_for_plot(algorithm_name)))
  plt.ylabel("Mean squared error")
  plt.xlabel("Number of gradient steps")
  plt.savefig("{}/{}-test-training-loss".format(directory_path, algorithm_name))
  plt.close()

  plt.plot(sine_wave_approximation_input_array, sine_wave_approximation_output_array)
  plt.scatter(sine_wave_input_array, sine_wave_output_array, marker="D", s=28, color="magenta")
  plt.plot(sine_wave_approximation_input_array, sine_wave_approximation_prediction_array[0], linestyle="-")
  plt.plot(sine_wave_approximation_input_array, sine_wave_approximation_prediction_array[1], linestyle="--")
  plt.plot(sine_wave_approximation_input_array, sine_wave_approximation_prediction_array[10], linestyle="-.")
  plt.title("{} sine wave approximation during testing".format(format_algorithm_name_for_plot(algorithm_name)))
  plt.ylabel("Sine wave output")
  plt.xlabel("Sine wave input")
  plt.legend(["ground truth", "pre-update", "one gradient step", "ten gradient steps", "data used for training"],
             prop={"size": 10})
  plt.savefig("{}/{}-test-sine-wave-approximation".format(directory_path, algorithm_name))
  plt.close()


def parse_input_arguments(baseline_training_iteration_choices=range(5000, 50001, 5000),
                          meta_training_iteration_choices=range(1000, 10001, 1000),
                          meta_batch_size_choices=range(5, 21, 5), batch_size_choices=range(5, 21, 5),
                          seed_choices=range(0, 21, 1)):
  parser = argparse.ArgumentParser(prog="run_experiment",
                                   usage="runs model-agnostic meta-learning supervised regression experiments")
  parser.add_argument("baseline_training_iterations", type=int, choices=baseline_training_iteration_choices,
                      help="number of training iterations for baseline")
  parser.add_argument("meta_training_iterations", type=int, choices=meta_training_iteration_choices,
                      help="number of meta-training iterations for model-agnostic meta-learning")
  parser.add_argument("meta_batch_size", type=int, choices=meta_batch_size_choices,
                      help="meta-training task batch size for model-agnostic meta-learning")
  parser.add_argument("batch_size", type=int, choices=batch_size_choices,
                      help="data batch size per task for training and testing")
  parser.add_argument("seed", type=int, choices=seed_choices,
                      help="seed value to control the randomness and get reproducible results")
  parser.add_argument("experimental_result_path", help="directory path to save the experimental results")

  input_arguments = parser.parse_args()

  return input_arguments
