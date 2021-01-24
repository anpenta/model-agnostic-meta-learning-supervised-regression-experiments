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

# Experiments Module
# Model-agnostic meta-learning supervised regression experiment functions.

import numpy as np
import torch

import sine_wave_task
import utility


def train_baseline_regressor(baseline_regressor, baseline_training_iterations, batch_size, torch_device):
  loss_array = np.zeros(baseline_training_iterations)
  for iteration in range(baseline_training_iterations):
    # Create a new task and compute sine wave input and output batches.
    task = sine_wave_task.SineWaveTask()
    sine_wave_input_array, sine_wave_output_array = task.sample_sine_wave_arrays(batch_size)
    sine_wave_input_batch = utility.transform_array_to_tensor(sine_wave_input_array, torch_device, torch.float).unsqueeze(1)
    sine_wave_output_batch = utility.transform_array_to_tensor(sine_wave_output_array, torch_device, torch.float).unsqueeze(1)

    # Compute sine wave prediction batch and update the model based on the loss.
    sine_wave_prediction_batch = baseline_regressor(sine_wave_input_batch)
    loss = baseline_regressor.loss_function(sine_wave_prediction_batch, sine_wave_output_batch)
    baseline_regressor.optimizer.zero_grad()
    loss.backward()
    baseline_regressor.optimizer.step()

    loss_array[iteration] = loss.item()

  return loss_array


def train_maml_regressor(maml_regressor, meta_training_iterations, meta_batch_size, batch_size, torch_device):
  meta_loss_array = np.zeros(meta_training_iterations)
  for iteration in range(meta_training_iterations):
    meta_loss = 0
    for _ in range(meta_batch_size):
      # Create a new task and compute sine wave input and output batches.
      task = sine_wave_task.SineWaveTask()
      sine_wave_input_array, sine_wave_output_array = task.sample_sine_wave_arrays(batch_size)
      sine_wave_input_batch = utility.transform_array_to_tensor(sine_wave_input_array, torch_device, torch.float).unsqueeze(1)
      sine_wave_output_batch = utility.transform_array_to_tensor(sine_wave_output_array, torch_device, torch.float).unsqueeze(1)

      # Compute sine wave prediction batch using a temporary weight list and update the temporary weights based on the loss.
      temporary_weight_list = [weight.clone() for weight in maml_regressor.parameters()]
      sine_wave_prediction_batch = maml_regressor.static_forward(sine_wave_input_batch, temporary_weight_list)
      loss = maml_regressor.loss_function(sine_wave_prediction_batch, sine_wave_output_batch)
      gradient = torch.autograd.grad(loss, temporary_weight_list)
      updated_temporary_weight_list = [weight - maml_regressor.learning_rate * derivative for weight, derivative in zip(temporary_weight_list, gradient)]

      # Compute sine wave input and output batches for meta-learning.
      meta_sine_wave_input_array, meta_sine_wave_output_array = task.sample_sine_wave_arrays(batch_size)
      meta_sine_wave_input_batch = utility.transform_array_to_tensor(meta_sine_wave_input_array, torch_device, torch.float).unsqueeze(1)
      meta_sine_wave_output_batch = utility.transform_array_to_tensor(meta_sine_wave_output_array, torch_device, torch.float).unsqueeze(1)

      # Compute sine wave prediction batch for meta-learning using the updated temporary weight list and accumulate the meta-loss.
      meta_sine_wave_prediction_batch = maml_regressor.static_forward(meta_sine_wave_input_batch, updated_temporary_weight_list)
      meta_loss += maml_regressor.loss_function(meta_sine_wave_prediction_batch, meta_sine_wave_output_batch) / meta_batch_size

    # Update the model-agnostic meta-learning regressor's weights based on the meta-loss.
    meta_gradient = torch.autograd.grad(meta_loss, list(maml_regressor.parameters()))
    for weight, derivative in zip(maml_regressor.parameters(), meta_gradient):
      weight.grad = derivative
    maml_regressor.optimizer.step()

    meta_loss_array[iteration] = meta_loss.item()

  return meta_loss_array


def test_regressors(trained_baseline_regressor, trained_maml_regressor, training_iterations, batch_size, torch_device):
  # Create a new task and compute sine wave approximation using mock input. Prepare sine wave approximation prediction
  # array to store the regressors' sine wave approximation predictions during training.
  task = sine_wave_task.SineWaveTask()
  sine_wave_approximation_input_array = np.linspace(-5, 5, 100)
  sine_wave_approximation_output_array = task.compute_sine_wave_output_array(sine_wave_approximation_input_array)
  sine_wave_approximation_input_batch = utility.transform_array_to_tensor(sine_wave_approximation_input_array, torch_device, torch.float).unsqueeze(1)
  sine_wave_approximation_prediction_array = np.zeros((2, training_iterations, 100))

  # Compute sine wave input and output batches.
  sine_wave_input_array, sine_wave_output_array = task.sample_sine_wave_arrays(batch_size)
  sine_wave_input_batch = utility.transform_array_to_tensor(sine_wave_input_array, torch_device, torch.float).unsqueeze(1)
  sine_wave_output_batch = utility.transform_array_to_tensor(sine_wave_output_array, torch_device, torch.float).unsqueeze(1)

  loss_array = np.zeros((2, training_iterations))
  for regressor_index, trained_regressor in enumerate((trained_baseline_regressor, trained_maml_regressor)):
    for iteration in range(training_iterations):
      # Compute and store sine wave approximation prediction batch.
      sine_wave_approximation_prediction_batch = trained_regressor(sine_wave_approximation_input_batch).squeeze(1)
      sine_wave_approximation_prediction_array[regressor_index][iteration] = utility.transform_tensor_to_array(sine_wave_approximation_prediction_batch)

      # Compute sine wave prediction batch and update the model based on the loss.
      sine_wave_prediction_batch = trained_regressor(sine_wave_input_batch)
      loss = trained_regressor.loss_function(sine_wave_prediction_batch, sine_wave_output_batch)
      trained_regressor.optimizer.zero_grad()
      loss.backward()
      trained_regressor.optimizer.step()

      loss_array[regressor_index][iteration] = loss.item()

  return (loss_array, sine_wave_approximation_prediction_array, sine_wave_input_array, sine_wave_output_array,
          sine_wave_approximation_input_array, sine_wave_approximation_output_array)
