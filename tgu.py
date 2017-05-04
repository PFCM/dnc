"""Tensor Gate Unit RNN core. Similar to an LSTM, but with more advanced
memory access semantics."""
import tensorflow as tf
import sonnet as snt


class CPBilinear(snt.AbstractModule):
  """Implements a bilinear tensor product between two vectors and a
  three tensor of weights in which the three-way tensor is represented in the
  CP decomposition. The main part of the tensor (forgetting about biases) is
  represented as three matrices A, B and C. For input vectors x and y this
  results in the following product:
    z = A^T(Bx . Cy)
  where '.' represents an elementwise product.

  If used, biases are treated separately by constructing a bias term
    Ux + Vy + b
  which falls out of the definition of the bilinear tensor product before
  taking the decomposition into account.
  """

  def __init__(self, output_size, rank, use_bias=True, name='cp_bilinear'):
    """
    Set up the bilinear product.

    Args:
      output_size (int): the size of the result vector.
      rank (int): the rank of the tensor decomposition.
      name (Optional[str]): the name of the module.
    """
    super(CPBilinear, self).__init__(name=name)

    with self._enter_variable_scope():
      self._input_embedding_x = snt.Linear(rank,
                                           use_bias=False,
                                           name='input_embedding_x')
      self._input_embedding_y = snt.Linear(rank,
                                           use_bias=False,
                                           name='input_embedding_y')
      self._output_projection = snt.Linear(output_size,
                                           use_bias=False,
                                           name='output_projection')
      if use_bias:
        self._bias_term = snt.Linear(output_size,
                                     use_bias=True,
                                     name='bias')
      else:
        self._bias_term = None

  def _build(self, input_x, input_y):
    """
    Connect the module to the graph.

    Args:
      input_x: the first input
      input_y: the second input

    Returns:
      result: `[batch, output_size]` batch of results.
    """
    embed_x = self._input_embedding_x(input_x)
    embed_y = self._input_embedding_y(input_y)

    result = self._output_projection(embed_x * embed_y)

    if self._bias_term is not None:
      result += self._bias_term(tf.concat([input_x, input_y], 1))

    return result


class TGU(snt.RNNCore):
  """
  Tensor gate unit RNN core.

  Implements the following for input x_t:
    z_t = W_z x_t
    f_t = \sigma(x_tW_fc_{t-1})  (tensor product)
    c_t = f_t . c_{t-1} + (1-f_t) . c_t
    h_t = \tau(x_tW_hc_t)  (tensor product)
  """

  def __init__(self, num_units, rank, name='tgu'):
    """
    Sets up the TGU.

    Args:
      num_units (int): the number of hidden units.
      rank (int): the rank of the two tensor decompositions.
      name (Optional[str]): name for the module.
    """
    super(TGU, self).__init__(name=name)
    self._hidden_size = num_units

    with self._enter_variable_scope():
      self._input_projection = snt.Linear(num_units,
                                          name='input_projection')
      self._forget_activations = CPBilinear(num_units,
                                            rank,
                                            use_bias=True,
                                            name='forget_gate')
      self._output_activations = CPBilinear(num_units,
                                            rank,
                                            use_bias=True,
                                            name='output')

  def _build(self, inputs, states):
    """
    Connect a single timestep of TGU computation.

    Args:
      inputs: input tensor for the current time step
      states: previous cell states

    Returns:
      output, state: outputs for the current timestep and states to carry over.
    """
    candidate = self._input_projection(inputs)
    forget_gate = tf.nn.sigmoid(self._forget_activations(inputs, states))

    new_states = forget_gate * states + (1.0 - forget_gate) * candidate

    outputs = tf.nn.tanh(self._output_activations(inputs, new_states))

    return outputs, new_states

  @property
  def state_size(self):
    """Returns the state size, with no batches."""
    return tf.TensorShape([self._hidden_size])

  @property
  def output_size(self):
    """Returns the output size, without batch dim"""
    return tf.TensorShape([self._hidden_size])
