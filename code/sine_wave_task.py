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

# Sine Wave Task Module
# Model of a sine wave task.

import numpy as np


class SineWaveTask:

  def __init__(self):
    self._sine_wave_amplitude = np.random.uniform(0.1, 5.0)
    self._sine_wave_phase = np.random.uniform(0.0, np.pi)

  def compute_sine_wave_output_array(self, sine_wave_input_array):
    return self._sine_wave_amplitude * np.sin(sine_wave_input_array - self._sine_wave_phase)

  def sample_sine_wave_arrays(self, batch_size):
    sine_wave_input_array = np.random.uniform(-5.0, 5.0, batch_size)
    sine_wave_output_array = self.compute_sine_wave_output_array(sine_wave_input_array)

    return sine_wave_input_array, sine_wave_output_array

  @property
  def sine_wave_amplitude(self):
    return self._sine_wave_amplitude

  @property
  def sine_wave_phase(self):
    return self._sine_wave_phase
