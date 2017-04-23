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
tf.flags.DEFINE_integer("num_epochs", 2,
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
    global_counter = tf.Variable(0, trainable=False)
    # For now the embedding is a matrix to be learned from scratch
    embedding_matrix = tf.get_variable(
        name="embedding_matrix",
        shape=[FLAGS.vocab_size, FLAGS.embedding_size],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())

    input_words = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.sentence_length])
    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)
    lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size)
    # Somehow lstm has a touple of states instead of just one.
    lstm_state = lstm.zero_state(FLAGS.batch_size, tf.float32)

    out_to_logit_w = tf.get_variable(
        name="output_weights",
        shape=[FLAGS.lstm_size, FLAGS.vocab_size],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
    out_to_logit_b = tf.Variable(tf.zeros([FLAGS.vocab_size]))

    probabilities = []
    loss = 0.0

    with tf.variable_scope("RNN"):
        for time_step in range(FLAGS.sentence_length):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            lstm_out, lstm_state = lstm(embedded_words[:, time_step, :],
                                        lstm_state)
            logits = tf.matmul(lstm_out, out_to_logit_w) + out_to_logit_b
            probabilities.append(tf.nn.softmax(logits))
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=input_words[:, time_step],
                logits=logits)

    perplexity = tf.exp(tf.reduce_mean(loss) / FLAGS.sentence_length)

    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate)
    gradients, v = zip(*optimiser.compute_gradients(loss))
    # TODO from the doc it looks as if we actually wanted to set use_norm to 10 instead. Confirm!
    # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
    train_op = optimiser.apply_gradients(zip(clipped_gradients, v),
                                         global_step=global_counter)

    init_op = tf.global_variables_initializer()

    # loop over training batches
    with tf.Session() as sess:
        sess.run(init_op)  # TODO add saver and restore if needed
        for data_train in batches_train:
            gc_, _, pp_ = sess.run([global_counter, train_op, perplexity],
                                   feed_dict={input_words: data_train})
            if gc_ % 100 == 0 and gc_ > 1:
                print(type(gc_))
                print(type(pp_))
                print("Current perplexity: %s" % pp_)


if __name__ == '__main__':
    tf.app.run()
