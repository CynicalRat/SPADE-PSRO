# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay buffer of fixed size with a FIFI replacement policy."""

import random
import torch

class ReplayBuffer(object):
    """ReplayBuffer of fixed size with a FIFO replacement policy.

    Stored transitions can be sampled uniformly.

    The underlying datastructure is a ring buffer, allowing 0(1) adding and
    sampling.
    """

    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = int(replay_buffer_capacity)
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        """Adds `element` to the buffer.

        If the buffer is full, the oldest element will be replaced.

        Args:
            element: data to be added to the buffer.
        """
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index] = element
            self._next_entry_index = (self._next_entry_index + 1) % self._replay_buffer_capacity

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
            num_samples: `int`, number of samples to draw.

        Returns:
            An iterable over `num_samples` random elements of the buffer.

        Raises:
            ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)))
        return random.sample(self._data, num_samples)
    
    def sample_sequences(self, batch_size, seq_len):
        assert len(self._data) >= seq_len
        starts = [random.randint(0, len(self._data)-seq_len) for _ in range(batch_size)]
        obs_batch, act_batch, rew_batch = [], [], []
        for s in starts:
            chunk = self._data[s : s+seq_len]
            obs_batch.append([t[0] for t in chunk])
            act_batch.append([t[1] for t in chunk])
            rew_batch.append([t[2] for t in chunk])
        return (
            torch.stack([torch.stack(x, dim=0) for x in obs_batch], dim=0),   # [B, T, obs_dim]
            torch.stack([torch.stack(x, dim=0) for x in act_batch], dim=0),   # [B, T, act_dim]
            torch.stack([torch.stack(x, dim=0) for x in rew_batch], dim=0),   # [B, T]
        )

    def reset(self):
        """Resets the contents of the replay buffer."""
        self._data = []
        self._next_entry_index = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)