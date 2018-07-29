"""Grid search training over a few parameters.

The PARAMATERS dict specifies the flags. They are all treated as
iterables and we loop over all combinations, the the correct values
and run the train function. Finally we tear everything down and start
again.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import string
import json
import itertools

import tensorflow as tf

import train


PARAMETERS = {
    # task specific
    'max_length': [10],
    'num_bits': [8],
    'max_repeats': [10],
    'sequence_length': [100, 200, 500, 1000, 10000],
    'task': ['copy_memory'],
    # bookkeeping
    'summary_interval': [200],
    'checkpoint_interval': [1000],
    'num_training_iterations': [50000],
    # (checkpoint_dir we'll do dynamically)
    # optimisation
    'learning_rate': [1e-3, 1e-2, 1e-4], #, 1e-3],
    'batch_size': [16],
    # model specifics
    'depth': [1],  # (we'll ignore this if use_dnc is True)
    'hidden_size': [128],  # override to 100 if use_dnc
    'use_dnc': [False], #, True],
    'controller_type': ['tguv2sigmoid', 'tgu', 'gru', 'lstm'],
    'memory_size': [32],
    'word_size': [16],
    'num_read_heads': [1],
    'max_grad_norm': [100]
}

CHECKPOINT_BASE = '/media/storage/dnc/runs/{}/big_grid'.format(PARAMETERS['task'][0])


FLAGS = tf.app.flags.FLAGS


def make_checkpoint_path(current_params):
  """makes a path for the checkpoints, given the current parameters."""
  # only use the flags that have been changed
  variable_param_names = [key for key in PARAMETERS
                          if len(PARAMETERS[key]) > 1]
  # make sure they're ordered deterministically
  variable_param_names = sorted(variable_param_names)
  variable_params = ['{}-{}'.format(name, current_params[name])
                     for name in variable_param_names]
  return os.path.join(CHECKPOINT_BASE, *variable_params)


def param_dicts():
  """Iterate through combinations of parameters, at each stage returning a full
  dict with a single value for every parameter."""
  keys = sorted([key for key in PARAMETERS])  # sort for determinism
  param_seqs = [PARAMETERS[key] for key in keys]
  for single_values in itertools.product(*param_seqs):
    yield {key: value for key, value in zip(keys, single_values)}


def dump_params(params, directory):
  """write out a text file showing the parameters for a specific run"""
  with open(os.path.join(directory, 'params.json'), 'w') as fpointer:
    json.dump(params, fpointer, indent=2)


def set_global_flags(params):
  """Sets attributes on the global FLAGS object which should hopefully carry
  through. Will raise errors if the parameter has not been defined somewhere"""
  for attr in params:
    if not hasattr(FLAGS, attr):
      raise AttributeError(
          'Attempting to set parameter {}, is not defined'.format(attr))
    setattr(FLAGS, attr, params[attr])


def run_search():
  """Run the search over the grid of parameters"""
  total_runs = 0
  for params in param_dicts():
    checkpoint_dir = make_checkpoint_path(params)

    # a few things to tidy up by hand
    if params.get('use_dnc', False):
      params['hidden_size'] = 100
      params['depth'] = 1
      # don't overdo the dncs
      if 'tgu' in params['controller_type']:
        params['controller_type'] = 'tguv2sigmoid'

    if params['task'] == 'repeat_copy':
      params['stop_threshold'] = 1.0  # really quite low
    elif params['task'] == 'variable_assignment':
      params['stop_threshold'] = 1e-3
      params['depth'] = 1  # for better comparision with ALSTM paper
      # params['num_training_iterations'] *= 5  # these are much faster

    for run in string.ascii_lowercase[:2]:
      # if the directory exists, it's a duplicate so skip it
      run_chkpt = os.path.join(checkpoint_dir, run)
      try:
        os.makedirs(run_chkpt)
        params['checkpoint_dir'] = run_chkpt
        print('-'*25)
        print('Beginning trial {}'.format(total_runs+1))
        print(run_chkpt)
        dump_params(params, run_chkpt)
        set_global_flags(params)
        # make sure the runs are separated appropriately
        with tf.Graph().as_default():
          train.main(None)

        total_runs += 1
      except OSError:
        print('skipping duplicate run')
      except KeyboardInterrupt:
        print('skipping interrupted run in a few seconds')
        time.sleep(5)

  print('search finished, ran {} trials'.format(total_runs))



if __name__ == '__main__':
  run_search()
