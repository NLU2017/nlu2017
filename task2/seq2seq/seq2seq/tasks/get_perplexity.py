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
"""
Task where both the input and output sequence are plain text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from pydoc import locate

import numpy as np

import tensorflow as tf
from tensorflow import gfile

from seq2seq.tasks.inference_task import InferenceTask, unbatch_dict

def _get_prediction_length(predictions_dict):
  """Returns the length of the prediction based on the index
  of the first SEQUENCE_END token.
  """
  tokens_iter = enumerate(predictions_dict["predicted_tokens"])
  return next(((i + 1) for i, _ in tokens_iter if _ == "SEQUENCE_END"),
              len(predictions_dict["predicted_tokens"]))


def _get_unk_mapping(filename):
  """Reads a file that specifies a mapping from source to target tokens.
  The file must contain lines of the form <source>\t<target>"

  Args:
    filename: path to the mapping file

  Returns:
    A dictionary that maps from source -> target tokens.
  """
  with gfile.GFile(filename, "r") as mapping_file:
    lines = mapping_file.readlines()
    mapping = dict([_.split("\t")[0:2] for _ in lines])
    mapping = {k.strip(): v.strip() for k, v in mapping.items()}
  return mapping


def _unk_replace(source_tokens,
                 predicted_tokens,
                 attention_scores,
                 mapping=None):
  """Replaces UNK tokens with tokens from the source or a
  provided mapping based on the attention scores.

  Args:
    source_tokens: A numpy array of strings.
    predicted_tokens: A numpy array of strings.
    attention_scores: A numeric numpy array
      of shape `[prediction_length, source_length]` that contains
      the attention scores.
    mapping: If not provided, an UNK token is replaced with the
      source token that has the highest attention score. If provided
      the token is insead replaced with `mapping[chosen_source_token]`.

  Returns:
    A new `predicted_tokens` array.
  """
  result = []
  for token, scores in zip(predicted_tokens, attention_scores):
    if token == "UNK":
      max_score_index = np.argmax(scores)
      chosen_source_token = source_tokens[max_score_index]
      new_target = chosen_source_token
      if mapping is not None and chosen_source_token in mapping:
        new_target = mapping[chosen_source_token]
      result.append(new_target)
    else:
      result.append(token)
  return np.array(result)

class GetPerplexity(InferenceTask):
  """Defines inference for tasks where both the input and output sequences
  are plain text.

  Params:
    delimiter: Character by which tokens are delimited. Defaults to space.
    unk_replace: If true, enable unknown token replacement based on attention
      scores.
    unk_mapping: If `unk_replace` is true, this can be the path to a file
      defining a dictionary to improve UNK token replacement. Refer to the
      documentation for more details.
    dump_attention_dir: Save attention scores and plots to this directory.
    dump_attention_no_plot: If true, only save attention scores, not
      attention plots.
    dump_beams: Write beam search debugging information to this file.
  """

  def __init__(self, params):
    super(GetPerplexity, self).__init__(params)
    self._unk_mapping = None

  @staticmethod
  def default_params():
    params = {}
    params.update({})
    return params

  def before_run(self, _run_context):
    fetches = {}
    fetches["losses"] = self._predictions["losses"]
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results
    for fetches in unbatch_dict(fetches_batch):
      # calculate perplexity
      losses = fetches['losses']
      perplexity = np.exp(sum(losses) / len(losses))
      print(perplexity)
