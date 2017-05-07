# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Character by character arithmetic task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import string
import numpy as np
import sonnet as snt
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations',
                                                           'target', 'mask'))


def masked_softmax_cross_entropy(logits,
                                 target,
                                 mask,
                                 time_average=False,
                                 log_prob_in_bits=False):
  """Adds ops to graph which compute the (scalar) NLL of the target sequence.

  The logits parametrize softmax distributions per time-step and
  per batch element, and irrelevant time/batch elements are masked out by the
  mask tensor.

  Args:
    logits: `Tensor` of activations.
    target: time-major integer `Tensor` of target.
    mask: time-major `Tensor` to be multiplied elementwise with cost T x B cost
        masking out irrelevant time-steps.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).

  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                        logits=logits)
  loss_time_batch = xent
  loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)

  batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

  if time_average:
    mask_count = tf.reduce_sum(mask, axis=0)
    loss_batch /= (mask_count + np.finfo(np.float32).eps)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss


def intstring_readable(data, batch_size, model_output=None,
                       whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    vocab: a dict of int->char to translate to something more readable.
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch
  """
  vocab = {i: i for i in xrange(10)}  # 0-9
  vocab[10] = ' '
  vocab[11] = '+'
  vocab[12] = '-'
  vocab[13] = '='
  def _readable(datum):
    return '+ ' + ''.join([vocab[x] for x in datum]) + ' +'

  obs_batch = data.observations
  targ_batch = data.target
  if model_output is not None:
    model_output = np.argmax(model_output, axis=2)

  iterate_over = xrange(batch_size) if whole_batch else xrange(1)

  batch_strings = []
  for batch_index in iterate_over:
    obs = obs_batch[:, batch_index]
    targ = targ_batch[:, batch_index]

    obs_channel_string = _readable(obs)
    targ_channel_string = _readable(targ)

    readable_obs = 'Observations:\n' + obs_channel_string
    readable_targ = 'Targets:\n' + targ_channel_string
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output[:, batch_index]
      output_string = _readable(output)
      strings.append('Model Output:\n' + output_string)

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)


class Addition(snt.AbstractModule):
  """Sequence data generator for the arithmetic task.
  Should be more or less as per https://openreview.net/pdf?id=BydARw9ex
  although with a few gaps in the description filled in in whatever way is
  easiest to implement.

  When called, an instance of this class will return a tuple of tensorflow ops
  (obs, targ, mask), representing an input sequence, target sequence, and
  binary mask. Each of these ops produces tensors whose first two dimensions
  represent sequence position and batch index respectively. The value in
  mask[t, b] is equal to 1 iff a prediction about targ[t, b, :] should be
  penalized and 0 otherwise.

  Input consist of two random integers uniformly sampled between -1e7 and 1e7
  (with one character per timestep), a plus symbol in between and final `=`
  symbols to make up the remaining length.

  The target is the sum of the two numbers, one character at a time.

  Input and output numbers are represented as one-hot encodings, with 4
  special characters: `=` to show the input has finished, `+` to separate the
  two numbers, `-` to show a number is negative and ` ` to pad all numbers to
  the same length. This means we have 14 input dimensions, but only 12 output
  dimensions (because we don't need to output `+` or `=`)

  An example sequence is shown below
  ```none
  Note: blank space means '0'

  time --------------------------------------->

                +------------------------------+
  mask:         |000000000000000000011111111111|
                +------------------------------+

                                       -7776010
                +------------------------------+
  target:       |                      1       |  '-'
                |                     1        |  ' '
                |                              |  '9'
                |                              |  '8'
                |                       111    |  '7'
                |                          1   |  '6'
                |                              |  '5'
                |                              |  '4'
                |                              |  '3'
                |                              |  '2'
                |                            1 |  '1'
                |                           1 1|  '0'
                +------------------------------+

                ( -9123842+  1347832===========)
                +------------------------------+
  observation:  |                   11111111111|  '='
                |         1                    |  '+'
                | 1                            |  '-'
                |1         11                  |  ' '
                |  1                           |  '9'
                |      1         1             |  '8'
                |               1              |  '7'
                |                              |  '6'
                |                              |  '5'
                |       1      1               |  '4'
                |     1       1   1            |  '3'
                |    1   1         1           |  '2'
                |   1        1                 |  '1'
                |                              |  '0'
                +------------------------------+
  ```

  All numbers are left padded with the ` ` character where necessary, so all
  sequences are exactly length 30. This is because we have 9 characters each
  for the input numbers (max 8 digits and a sign or a space), 1 `+`, a
  final `=` and 10 characters for the output (a sign and at most 9 digits).
  9 + 1 + 9 + 1 + 10 = 30
  """

  def __init__(
      self,
      batch_size,
      log_prob_in_bits=False,
      name='addition'):
    """Creates an instance of Addition task.

    Args:
      batch_size: Minibatch size per realization.
      log_prob_in_bits: By default, log probabilities are expressed in units of
        nats. If true, express log probabilities in bits.
      name: A name for the generator instance (for name scope purposes).
    """
    super(Addition, self).__init__(name)

    self._batch_size = batch_size
    self._log_prob_in_bits = log_prob_in_bits
    # make a sparse matrix containing one hots corresponding to the ascii code
    # of the characters we need
    with self._enter_variable_scope():
      indices = [[ord(char), i] for i, char in enumerate('0123456789 -+=')]
      self._one_hots = tf.sparse_to_dense(indices, [128, 14], tf.ones([14]))

  @property
  def log_prob_in_bits(self):
    return self._log_prob_in_bits

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def target_size(self):
    return 12

  def _build(self):
    """Implements build method which adds ops to graph."""
    input_a = tf.random_uniform([self._batch_size],
                                minval=-10000000,
                                maxval=10000000,
                                dtype=tf.int32)
    input_b = tf.random_uniform([self._batch_size],
                                minval=-10000000,
                                maxval=10000000,
                                dtype=tf.int32)
    target = input_a + input_b
    plus = tf.constant('+')
    equals = tf.constant('='*9)

    str_obs = tf.string_join([tf.as_string(input_a,
                                           width=9),
                              plus,
                              tf.as_string(input_b,
                                           width=9),
                              equals])
    # decode then cast so we only do one byte at a time
    int_obs = tf.cast(tf.decode_raw(str_obs, tf.uint8), tf.int32)
    # now map the ascii codes down to appropriately sized one-hots
    obs = tf.gather(self._one_hots, int_obs)

    # and the target
    str_targ = tf.as_string(target,
                            width=30)  # left pad by default
    int_targ = tf.cast(tf.decode_raw(str_targ, tf.uint8), tf.int32)
    targ = tf.gather(self._one_hots, int_targ)[:, :, :13]

    # the mask is always the same
    mask = tf.concat([tf.zeros([20, self._batch_size]),
                      tf.ones([10, self._batch_size])])

    return DatasetTensors(obs, targ, mask)

  def cost(self, logits, targ, mask):
    return masked_softmax_cross_entropy(
        logits,
        targ,
        mask,
        time_average=False,
        log_prob_in_bits=self.log_prob_in_bits)

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    return intstring_readable(data, self.batch_size, model_output, whole_batch)
