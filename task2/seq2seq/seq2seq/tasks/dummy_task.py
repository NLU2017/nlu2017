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






class DummyTask(InferenceTask):
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
    super(DummyTask, self).__init__(params)
    self._counter = 0


  @staticmethod
  def default_params():
    params = {}
    params.update({
        "delimiter": " "
    })
    return params

  def before_run(self, _run_context):
    fetches = {}
    fetches["predicted_tokens"] = self._predictions["predicted_tokens"]
    fetches["predicted_ids"] = self._predictions["predicted_ids"]
    fetches["logits"] = self._predictions["logits"]
    fetches["labels.target_len"] = self._predictions["labels.target_len"]
    fetches["losses"] = self._predictions["losses"]

    return tf.train.SessionRunArgs(fetches)


  def after_run(self, _run_context, run_values):
    fetches_batch = run_values.results

    for fetches in unbatch_dict(fetches_batch):
      # Convert to unicode
      fetches["predicted_tokens"] = np.char.decode(fetches["predicted_tokens"].astype("S"), "utf-8")
      predicted_tokens = fetches["predicted_tokens"]
      predicted_ids = fetches["predicted_ids"]


      target_len = fetches["labels.target_len"]
      logits = fetches["logits"]
      print("logits shape {}".format(logits.shape))
      #print(logits)

      print()
      print("target lengths")
      print(target_len)
      print()
      sent = self.params["delimiter"].join(predicted_tokens).split(
          "SEQUENCE_END")[0]


      print("counter {}".format(self._counter))
      print("size of predicted tokens {}".format(len(predicted_tokens)))
      print("predicted ids: {}".format(predicted_ids))
      self._counter += 1

      print(sent)
      losses = fetches["losses"]
      print("---------------LOSSES----------------")
      print(losses)
      print("-----prediction dict----")
      for d in self._predictions.items():
        print(d[0] + "  {}".format(d[1]))
