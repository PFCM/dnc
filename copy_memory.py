"""A repeat copy task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import string
import numpy as np
import sonnet as snt
import tensorflow as tf

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations',
                                                           'target',
                                                           'mask'))


def sequence_softmax_cross_entropy(logits,
                                   target,
                                   time_average=False,
                                   log_prob_in_bits=False):
  """Adds ops to the graph which compute the (scalar) NLL of a sequence of
  of tokens from some vocabulary.


  Args:
    logits: `Tensor` of activations for which softmax(`logits`) gives the
      likelihoods assigned to each token in the vocabulary for each timestep.
      Should be time-major (ie. shape `[time, batch, vocab_size]`)
    target: time-major `Tensor` of target integers.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).

  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=target)

  if time_average:
    loss = tf.reduce_mean(xent, axis=0)
  else:
    loss = tf.reduce_sum(xent, axis=0)
  # mean over batch
  loss = tf.reduce_mean(loss)
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss


def sequence_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.
  Doesn't work with a vocab of more than 26 (we just lookup ascii letters).

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch
  """

  def _lookup(number):
    if number > 0:
      return string.ascii_uppercase[number-1]
    return ':'

  def _readable(datum):
    return '+' + ''.join(['-' if x == 0 else _lookup(x) for x in datum]) + '+'

  obs_batch = data.observations
  obs_batch[-10] = -1
  obs_batch[:10] += 1
  targ_batch = data.target

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


class CopyMemory(snt.AbstractModule):
  """Generator of data for the copy-memory task. For each instance, a sequence
  of 10 tokens from a vocabulary of 8 possible symbols is generated. We then
  append T-1 blank symbols followed by a special 'go' symbol and finally 10
  more blanks.

  The target is for the network to produce blanks until the 'go' symbol and
  then recall the original sequence of 10 tokens, starting the time step after
  the 'go' symbol. To solve the problem it therefore has to remember the
  sequence for T time steps.

  For each item in each batch the sequence is created by sampling IID uniformly
  at random 10 items from a vocabulary of eight. Inputs are presented as
  one-hot vectors (except for the blank symbol which gets the all-zero vector)
  although for efficiency we leave the targets as integers. The inputs are
  therefore 9 dimensional and so should the inputs.

  To visualise this:

  ```none
  Note: blank space represents 0.
                           |----T-----|
  time --------------------|----------|------------>

                +--------------------------------+
  target:       |                      2547315868|
                +--------------------------------+

                +---------------------------------+
  observation:  |0000010000            0          |
                |1000000000            0          |
                |0000100000            0          |
                |0010000000            0          |
                |0100001000            0          |
                |0000000010            0          |
                |0001000000            0          |
                |0000000101            0          |
                |0000000000            1          | 'go' channel.
                +---------------------------------+
  ```

  In the observation we use a vector of all zeros to represent blank space and
  the final channel to represent the 'go' symbol. However, the network never
  needs to output the 'go' symbol but it does need to output the blank symbol,
  so the network needs 9 outputs, the first of which (after softmax) gives the
  likelihood of the blank symbol and the remaining 8 the eight symbols in the
  vocabulary.
  """

  def __init__(
      self,
      num_symbols=8,
      batch_size=1,
      num_blanks=100,
      sequence_length=10,
      log_prob_in_bits=False,
      time_average_cost=False,
      name='copy_memory',):
    """Creates an instance of RepeatCopy task.

    Args:
      name: A name for the generator instance (for name scope purposes).
      num_symbols: The number of symbols in the vocabulary used to build the
          sequences. The default (8) appears to be by far the most common.
      batch_size: Minibatch size per realization.
      num_blanks: The number of symbols inserted after the generated sequence.
          Increasing this is the usual way to make the problem harder.
      sequence_length: the length of the random sequence the network has to
          recall. This is usually 10.
      log_prob_in_bits: By default, log probabilities are expressed in units of
          nats. If true, express log probabilities in bits.
      time_average_cost: If true, the cost at each time step will be
          divided by the `true`, sequence length, the number of non-masked time
          steps, in each sequence before any subsequent reduction over the time
          and batch dimensions.
    """
    super(CopyMemory, self).__init__(name)

    self._batch_size = batch_size
    self._num_symbols = num_symbols
    self._num_blanks = num_blanks
    self._sequence_length = sequence_length
    self._log_prob_in_bits = log_prob_in_bits
    self._time_average_cost = time_average_cost

  @property
  def time_average_cost(self):
    return self._time_average_cost

  @property
  def log_prob_in_bits(self):
    return self._log_prob_in_bits

  @property
  def num_symbols(self):
    """The dimensionality of each random binary vector in a pattern."""
    return self._num_symbols

  @property
  def sequence_length(self):
    """The size of the sequence the RNN must recall."""
    return self._sequence_length

  @property
  def num_blanks(self):
    """The number of symbols after the end of the sequence before the RNN must
    start producing output."""
    return self._num_blanks

  @property
  def total_length(self):
    """The total size of the sequences"""
    return self._num_blanks + 2*self._sequence_length

  @property
  def target_size(self):
    """The dimensionality of the target tensor."""
    return self._num_symbols + 1

  @property
  def batch_size(self):
    return self._batch_size

  def _build(self):
    """Implements build method which adds ops to graph."""
    # should be able to do the lot in tensorflow...
    sequences = tf.random_uniform([self.sequence_length, self.batch_size],
                                  dtype=tf.int32,
                                  minval=0,
                                  maxval=self.num_symbols)
    one_hot_sequences = tf.one_hot(sequences, depth=self.num_symbols+1)
    middle_blanks = tf.zeros([self.num_blanks,
                              self.batch_size,
                              self.num_symbols+1])
    go_symbol = tf.one_hot([[self.num_symbols] * self.batch_size],
                           self.num_symbols+1)
    end_blanks = tf.zeros([self.sequence_length-1,
                           self.batch_size,
                           self.num_symbols+1])
    observation = tf.concat([one_hot_sequences,
                             middle_blanks,
                             go_symbol,
                             end_blanks],
                            axis=0)
    target = tf.concat([tf.zeros([self.sequence_length + self.num_blanks,
                                  self.batch_size],
                                 dtype=tf.int32),
                        sequences+1],
                       axis=0)

    return DatasetTensors(observation, target, tf.no_op())

  def cost(self, logits, targ, _):
    return sequence_softmax_cross_entropy(
        logits,
        targ,
        time_average=self.time_average_cost,
        log_prob_in_bits=self.log_prob_in_bits)

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    obs = data.observations
    int_obs = np.argmax(obs, axis=2)
    data = data._replace(observations=int_obs)
    if model_output is not None:
      model_output = np.argmax(model_output, axis=2)
    return sequence_readable(data, self.batch_size, model_output, whole_batch)
