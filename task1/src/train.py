import time
import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary
import numpy as np
import csv

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1,
                      "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_string("eval_file_path", "../data/sentences.eval",
                       "Path to the validation data")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the input sentences")
tf.flags.DEFINE_integer("vocab_size", 20000,
                        "Number of words in the vocabulary")
tf.flags.DEFINE_string("output_dir", "../data",
                       "Directory to store the results")
tf.flags.DEFINE_string("embedding", "../data/wordembeddings-dim100.word2vec",
                       "Path to the embedding file (space separated)")

# Model parameters
tf.flags.DEFINE_integer("lstm_size", 512, "Length of the hidden state")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimension of the embedding")
tf.flags.DEFINE_string("task", "C", "Task to be solved")
tf.flags.DEFINE_integer("intermediate_size", 512,
                        "Dimension of down-projection in task C")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1,
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
    train_loader = DataLoader(FLAGS.train_file_path,
                              vocabulary, do_shuffle=True)
    batches_train = train_loader.batch_iterator(FLAGS.num_epochs,
                                                FLAGS.batch_size)

    # load validation data
    eval_loader = DataLoader(FLAGS.eval_file_path,
                             vocabulary, do_shuffle=False)
    batches_eval = eval_loader.batch_iterator(num_epochs=1, batch_size=1000)

    # Create the graph
    global_counter = tf.Variable(0, trainable=False)
    if FLAGS.task == "A":
        # For a) the embedding is a matrix to be learned from scratch
        embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            shape=[FLAGS.vocab_size, FLAGS.embedding_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    else:
        # For simplicity we do not use the code given but use the given
        # embeddings directly.
        # TODO confirm that this is okay for the tutors.
        keys = []
        emb = []
        ext_emb = csv.reader(open(FLAGS.embedding), delimiter=' ')
        for line in ext_emb:
            keys += [line[0]]
            emb += [list(map(float, line[1:]))]
        given_emb = dict(zip(keys, emb))
        external_embedding = np.zeros(shape=(FLAGS.vocab_size,
                                             FLAGS.embedding_size))
        for k, v in vocabulary.get_vocabulary_as_dict().items():
            try:
                external_embedding[v, :] = given_emb[k]
            except KeyError:
                print("Unmached: %s" % k)
                external_embedding[v, :] = \
                    np.random.uniform(low=-0.25,
                                      high=0.25,
                                      size=FLAGS.embedding_size)
        embedding_matrix = tf.Variable(external_embedding, dtype=tf.float32)

    input_words = tf.placeholder(tf.int32, [None, FLAGS.sentence_length])
    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)
    lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size)
    # Somehow lstm has a touple of states instead of just one.
    # We learn sensible initial states as well. As I am unsure about whether
    # they are essentially the same, we train them seperately
    # TODO clarify what the two initial states mean
    lstm_zero_state_1 = \
        tf.get_variable("zero_state_1",
                        shape=[1, FLAGS.lstm_size],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
    lstm_zero_state_2 = \
        tf.get_variable("zero_state_",
                        shape=[1, FLAGS.lstm_size],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

    lstm_state = (tf.tile(lstm_zero_state_1, [tf.shape(input_words)[0], 1]),
                  tf.tile(lstm_zero_state_2, [tf.shape(input_words)[0], 1]))

    if not FLAGS.task == "C":
        out_to_logit_w = tf.get_variable(
            name="output_weights",
            shape=[FLAGS.lstm_size, FLAGS.vocab_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        out_to_logit_b = tf.Variable(tf.zeros([FLAGS.vocab_size]))
    else:
        inter_w = tf.get_variable("intermediate_weights",
                                  shape=[FLAGS.lstm_size,
                                         FLAGS.intermediate_size],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        inter_b = tf.get_variable("intermediate_biases",
                                  shape=[FLAGS.intermediate_size])

        out_to_logit_w = tf.get_variable(
            name="output_weights",
            shape=[FLAGS.intermediate_size, FLAGS.vocab_size],
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
            if not FLAGS.task == "C":
                logits = tf.matmul(lstm_out, out_to_logit_w) + out_to_logit_b
            else:
                logits = tf.matmul(tf.matmul(lstm_out, inter_w) + inter_b,
                                   out_to_logit_w) + out_to_logit_b
            probabilities.append(tf.nn.softmax(logits))
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=input_words[:, time_step],
                logits=logits)

    # TODO Confirm that the elementwise crossentropy is -p(w_t|w_1,...,w_{t-1})
    perplexity = tf.pow(2, tf.reduce_mean(loss) / FLAGS.sentence_length)

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
        print("Initialising")
        sess.run(init_op)  # TODO add saver and restore if needed
        print("Start running the training")
        for data_train in batches_train:
            gc_, _, pp_ = sess.run([global_counter, train_op, perplexity],
                                   feed_dict={input_words: data_train})
            if gc_ % 100 == 0:
                print("Iteration %s: Perplexity is %s" % (gc_, pp_))

        print("Starting validation")
        out_pp = []
        for data_eval in batches_eval:
            out_pp += sess.run([perplexity])
        # TODO find out group number
        np.savetext(
            FLAGS.output_dir + "/groupXX.perplexity" + FLAGS.task,
            np.array(out_pp), delimiter=',')


if __name__ == '__main__':
    tf.app.run()
