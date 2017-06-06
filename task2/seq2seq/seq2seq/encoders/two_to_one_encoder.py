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

class TwoToOneEcnoder(BidirectionalRNNEncoder):
  """
  A bidirectional RNN encoder. Uses the same cell for both the
  forward and backward RNN. Stacking should be performed as part of
  the cell.

  Args:
    cell: An instance of tf.contrib.rnn.RNNCell
    name: A name for the encoder
  """

  def __init__(self, params, mode, name="twotoone_rnn_encoder"):
    super(BidirectionalRNNEncoder, self).__init__(params, mode, name)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)
    self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

  @staticmethod
  def default_params():
    return {
        "rnn_cell": _default_rnn_cell_params(),
        "rnn_cell": _default_rnn_cell_params(),
        "init_scale": 0.04,
    }

  def encode(self, inputs, sequence_length, **kwargs):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))


    with tf.variable_scope("reader_1"):
        cell_fw0 = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw0 = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        print("inputs {}".format(inputs.get_shape()))
        outputs0, states0 = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw0,
            cell_bw=cell_bw0,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

    with tf.variable_scope("reader_2"):
        cell_fw1 = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw1 = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        print("seconds cell allright")
        outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw1,
            cell_bw=cell_bw1,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

    print("after bidirectional_dynamic-----------------------------------------------------------------------------------------------------------------------------------")
    print(states0)
    print(states1)
    outputs0_concat = tf.concat(outputs0, 2)
    outputs1_concat = tf.concat(outputs1, 2)
    outputs_concat = tf.concat([outputs0_concat, outputs1_concat], 2)

    # states_fw = tf.concat((states0[0], states1[0]), 2)
    # states_bw = tf.concat((states0[1], states1[1]), 2)

    states = (tf.contrib.rnn.LSTMStateTuple(
        c=tf.concat([states0[0][0], states1[0][0]], 1),
        h=tf.concat([states0[0][1], states1[0][1]], 1)),
          tf.contrib.rnn.LSTMStateTuple(
              c=tf.concat([states0[1][0], states1[1][0]], 1),
              h=tf.concat([states0[1][1], states1[1][1]], 1))
    )

    print("DEBUG: ---------------------------------------------------------------------------------------------------------------------------------------output and state format")
    print(outputs_concat)
    print(states)

    return EncoderOutput(
        outputs=outputs_concat,
        final_state=states,
        attention_values=outputs_concat,
        attention_values_length=sequence_length)

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
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

    with tf.variable_scope("reader_2"):
        outputs2, state2 = tf.nn.dynamic_rnn(
            cell=cell2,
            inputs=inputs,
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