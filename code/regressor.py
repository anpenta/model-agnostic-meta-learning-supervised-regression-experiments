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

# Regressor Module
# Model of a neural network regressor.

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Regressor(nn.Module):

  def __init__(self, learning_rate, torch_device):
    super().__init__()
    self._hidden_1 = nn.Linear(1, 40)
    self._hidden_2 = nn.Linear(40, 40)
    self._output = nn.Linear(40, 1)

    self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self._learning_rate = learning_rate
    self._loss_function = nn.MSELoss()
    self.to(torch_device)

  # The shape of the input to the forward function should be batch size by one.
  def forward(self, input_batch):
    hidden_batch = F.relu(self._hidden_1(input_batch))
    hidden_batch = F.relu(self._hidden_2(hidden_batch))
    prediction_batch = self._output(hidden_batch)

    return prediction_batch

  @staticmethod
  def static_forward(input_batch, weight_list):
    # Compute a forward pass using the given weight list.
    hidden_batch = F.relu(F.linear(input_batch, weight_list[0], weight_list[1]))
    hidden_batch = F.relu(F.linear(hidden_batch, weight_list[2], weight_list[3]))
    prediction_batch = F.linear(hidden_batch, weight_list[4], weight_list[5])

    return prediction_batch

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def learning_rate(self):
    return self._learning_rate

  @property
  def loss_function(self):
    return self._loss_function
