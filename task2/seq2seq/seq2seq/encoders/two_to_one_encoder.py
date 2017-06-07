from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.encoders.rnn_encoder import BidirectionalRNNEncoder
from seq2seq.training import utils as training_utils


def _default_rnn_cell_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "cell_class": "BasicLSTMCell",
      "cell_params": {
          "num_units": 128
      },
      "dropout_input_keep_prob": 1.0,
      "dropout_output_keep_prob": 1.0,
      "num_layers": 1,
      "residual_connections": False,
      "residual_combiner": "add",
      "residual_dense": False
  }


def _toggle_dropout(cell_params, mode):
  """Disables dropout during eval/inference mode
  """
  cell_params = copy.deepcopy(cell_params)
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    cell_params["dropout_input_keep_prob"] = 1.0
    cell_params["dropout_output_keep_prob"] = 1.0
  return cell_params


class TwoToOneEncoder(Encoder):
  def __init__(self, params, mode, name="forward_rnn_encoder"):
    super(TwoToOneEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def encode(self, inputs, sequence_length, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    cell1 = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    cell2 = training_utils.get_rnn_cell(**self.params["rnn_cell"])

    with tf.variable_scope("reader_1"):
        outputs1, state1 = tf.nn.dynamic_rnn(
            cell=cell1,
            inputs=inputs[0],
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

    with tf.variable_scope("reader_2"):
        outputs2, state2 = tf.nn.dynamic_rnn(
            cell=cell2,
            inputs=inputs[1],
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

    outputs = tf.concat([outputs1, outputs2], 2)
    state = tf.contrib.rnn.LSTMStateTuple(
        c=tf.concat([state1[0], state2[0]], 1),
        h=tf.concat([state1[1], state2[1]], 1))

    print("Checking outputs and states size")
    print(outputs)
    print(state)

    return EncoderOutput(
        outputs=outputs,
        final_state=state,
        attention_values=outputs,
        attention_values_length=sequence_length)