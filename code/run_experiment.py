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

# Run Experiment Module
# Module to run model-agnostic meta-learning supervised regression experiments.

import torch

import experiments
import regressor
import utility

if __name__ == "__main__":
  input_arguments = utility.parse_input_arguments()

  utility.control_randomness(input_arguments.seed)
  torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Create and train baseline regressor.
  baseline_regressor = regressor.Regressor(0.01, torch_device)
  experiments.train_baseline_regressor(baseline_regressor, input_arguments.baseline_training_iterations,
                                       input_arguments.batch_size, torch_device)

  # Create and train model-agnostic meta-learning regressor.
  maml_regressor = regressor.Regressor(0.01, torch_device)
  experiments.train_maml_regressor(maml_regressor, input_arguments.meta_training_iterations,
                                   input_arguments.meta_batch_size, input_arguments.batch_size, torch_device)

  # Evaluate the trained regressors, and create and save the test result plots.
  test_results = experiments.test_regressors(baseline_regressor, maml_regressor, 100, input_arguments.batch_size,
                                             torch_device)
  utility.create_and_save_test_result_plots(input_arguments.experimental_result_path, "baseline",
                                            test_results[0][0], test_results[1][0], *test_results[2:])
  utility.create_and_save_test_result_plots(input_arguments.experimental_result_path, "maml",
                                            test_results[0][1], test_results[1][1], *test_results[2:])
