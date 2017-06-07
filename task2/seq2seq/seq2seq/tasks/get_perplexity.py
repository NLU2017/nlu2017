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
      # Convert to unicode
      fetches["losses"] = np.char.decode(
          fetches["losses"].astype("S"), "utf-8")
      losses_ = fetches["losses"]

      # If we're using beam search we take the first beam
      if np.ndim(losses_) > 1:
        losses_ = losses_[:, 0]

      sent = self.params["delimiter"].join(losses_).split("\n")[0]

      sent = sent.strip()

      print(sent)
