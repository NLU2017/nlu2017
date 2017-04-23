import time
import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary


## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1,
                      "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_string("data_location", "../data/",
                       "Path to intermediate results")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the input sentences")
tf.flags.DEFINE_integer("vocab_size", 20000, "Number of words in the vocabulary")

# Model parameters
tf.flags.DEFINE_integer("lstm_size", 512, "Length of the hidden state")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimension of the embedding")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5,
                        "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("log_dir", "../runs/",
                       "Output directory (default: '../runs/')")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of the optimizer")
# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
FLAGS.model_dir = os.path.abspath(os.path.join(FLAGS.log_dir, timestamp))
print("Writing to {}\n".format(FLAGS.model_dir))


def main(unused_argv):
    # extract the vocabulary from training sentendes
    vocabulary = Vocabulary()
    vocabulary.load_file(FLAGS.train_file_path)

    # load training data
    train_loader = DataLoader(FLAGS.train_file_path, FLAGS.data_location,
                              vocabulary, do_shuffle=True)
    batches_train = train_loader.batch_iterator(FLAGS.num_epochs,
                                                FLAGS.batch_size)

    # Create the graph
    embedding_matrix = tf.random_uniform([FLAGS.vocab_size,
                                          FLAGS.embedding_size],
                                         -1.0 / tf.sqrt(tf.to_float(FLAGS.vocab_size)),
                                         1.0 / tf.sqrt(tf.to_float(FLAGS.vocab_size)))
    input_words = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.sentence_length])
    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)

    lstm_state = tf.zeros([FLAGS.batch_size, FLAGS.lstm_size])
    lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size)

    out_to_logit_w = tf.random_uniform(
        [FLAGS.lstm_size,
         FLAGS.vocab_size],
        -1.0/tf.sqrt(tf.to_float(FLAGS.lstm_size)),
        1.0/tf.sqrt(tf.to_float(FLAGS.lstm_size)))
    out_to_logit_b = tf.zeros([FLAGS.vocab_size])

    probabilities = []
    loss = 0.0

    with tf.variable_scope("RNN"):
        for time_step in range(FLAGS.sentence_length):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            a = embedded_words[:, time_step, :]
            b = lstm_state
            lstm(a, b)
            # TODO the next line does not run. I fail to see why.
            lstm_out, lstm_state = lstm(embedded_words[:, time_step, :], lstm_state)
            logits = tf.matmul(lstm_out, out_to_logit_w) + out_to_logit_b
            probabilities.append(tf.nn.softmax(logits))
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, input_words[:, time_step])

    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_vars = tf.trainable_variables()
    # TODO from the doc it looks as if we actually wanted to set use_norm to 10 instead. Confirm!
    # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
    clipped_gradients = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 10)
    train_op = optimiser.apply_gradients(zip(clipped_gradients, train_vars))

    # loop over training batches
    for data_train in batches_train:
        pass


if __name__ == '__main__':
    tf.app.run()
