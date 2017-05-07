import pickle
import time
import os
import tensorflow as tf
from utils import Vocabulary, DataLoader, clean_and_cut_sentences
import numpy as np
import csv
from load_embedding import load_embedding

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1,
                      "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_string("eval_file_path", "../data/sentences_test",
                       "Path to the validation data")
tf.flags.DEFINE_string("generate_file_path", "../data/sentences.continuation",
                       "Source file for incomplete sentences")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the input sentences")
tf.flags.DEFINE_integer("vocab_size", 20000,
                        "Number of words in the vocabulary")
tf.flags.DEFINE_string("output_dir", "../data",
                       "Directory to store the results")
tf.flags.DEFINE_string("embedding", "../data/wordembeddings-dim100.word2vec",
                       "Path to the embedding file (space separated)")
#phases
tf.flags.DEFINE_boolean("do_train", True, "Perform training")
tf.flags.DEFINE_boolean("do_eval", True, "Perform evaluation")
tf.flags.DEFINE_boolean("do_generate", True, "Perform generation")

# Model parameters
tf.flags.DEFINE_integer("lstm_size", 512, "Length of the hidden state")
tf.flags.DEFINE_integer("intermediate_size", 512,
                        "Dimension of down-projection in task C")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimension of the embedding")
tf.flags.DEFINE_string("task", "A", "Task to be solved")
tf.flags.DEFINE_string("model_name", "A_Vanilla", "Name the model")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3,
                        "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("log_dir", "../runs/",
                       "Output directory (default: '../runs/')")
tf.flags.DEFINE_float("learning_rate", 0.01,
                      "Inital learning rate of the optimizer")
tf.flags.DEFINE_integer("hlave_lr_every", 30000,
                        "Every n steps the learning rate is halved")
tf.flags.DEFINE_integer("no_output_before_n", 500,
                        "Supress the first outputs, because of strong changes")
tf.flags.DEFINE_float("lambda_l2", 0.00000000001, "Strength of L2 normalization")
tf.flags.DEFINE_boolean("force_init", False,
                        "Whether to always start training from scratch")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Create a unique output directory for this experiment.
FLAGS.model_dir = os.path.abspath(
    os.path.join(FLAGS.log_dir, FLAGS.model_name))
print("Writing to {}\n".format(FLAGS.model_dir))
os.makedirs(FLAGS.model_dir, exist_ok=True)


def main(unused_argv):
    # Initialise learning rate
    eff_rate = FLAGS.learning_rate

    # extract the vocabulary from training sentendes
    with open("vocabulary.pickle", "rb") as f:
        vocabulary = pickle.load(f)

    # load training data
    if FLAGS.do_train:
        train_loader = DataLoader(FLAGS.train_file_path,
                                  vocabulary, do_shuffle=True)
        batches_train = train_loader.batch_iterator(FLAGS.num_epochs,
                                                    FLAGS.batch_size)

    # load validation data
    if FLAGS.do_eval:
        eval_loader = DataLoader(FLAGS.eval_file_path,
                                 vocabulary, do_shuffle=False)
        batches_eval = eval_loader.batch_iterator(num_epochs=1,
                                                  batch_size=1000)

    # Load continuation data
    if FLAGS.do_generate:
        gen_loader = DataLoader(FLAGS.generate_file_path,
                                vocabulary, do_shuffle=False, is_partial=True)
        batches_gen = gen_loader.batch_iterator(num_epochs=1, batch_size=1000)

    # Create the graph
    global_counter = tf.Variable(0, trainable=False)
    input_words = tf.placeholder(tf.int32, [None, FLAGS.sentence_length])
    # add to collection for usage from restored model
    tf.add_to_collection("input_words", input_words)

    embedding_matrix = tf.get_variable(
        name="embedding_matrix",
        shape=[FLAGS.vocab_size, FLAGS.embedding_size],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())



    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)

    # RNN graph
    lstm = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size)
    # Somehow lstm has a touple of states instead of just one.
    # We learn sensible initial states as well.

    # The states seem to correspond to the LSTM cell state and the hidden layer output
    # see http://stackoverflow.com/questions/41789133/c-state-and-m-state-in-tensorflow-lstm
    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
    lstm_zero_c = \
        tf.get_variable("zero_state_c",
                        shape=[1, FLAGS.lstm_size],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
    lstm_zero_h = \
        tf.get_variable("zero_state_h",
                        shape=[1, FLAGS.lstm_size],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

    lstm_state = (tf.tile(lstm_zero_c, [tf.shape(input_words)[0], 1]),
                  tf.tile(lstm_zero_h, [tf.shape(input_words)[0], 1]))
    lstm_state_g = (tf.tile(lstm_zero_c, [tf.shape(input_words)[0], 1]),
                    tf.tile(lstm_zero_h, [tf.shape(input_words)[0], 1]))

    if not FLAGS.task == "C":
        out_to_logit_w = tf.get_variable(
            name="output_weights",
            shape=[FLAGS.lstm_size, FLAGS.vocab_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        out_to_logit_b = tf.get_variable("output_bias",
                                         shape=[FLAGS.vocab_size])
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
        out_to_logit_b = tf.get_variable("output_bias",
                                         shape=[FLAGS.vocab_size])

    # initialize
    lstm_outputs = []
    # add summaries for tensorboard

    with tf.variable_scope("RNN"):
        for time_step in range(FLAGS.sentence_length):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            lstm_out, lstm_state = lstm(embedded_words[:, time_step, :],
                                        lstm_state)

            lstm_outputs.append(lstm_out)

    output = tf.concat(axis=0, values=lstm_outputs)

    if not FLAGS.task == "C":
        logits = tf.matmul(output, out_to_logit_w) + out_to_logit_b
        l2_loss = tf.nn.l2_loss(out_to_logit_w) * FLAGS.lambda_l2
    else:
        logits = tf.matmul(tf.matmul(output, inter_w) + inter_b,
                           out_to_logit_w) + out_to_logit_b
        # l2_loss = (tf.nn.l2_loss(out_to_logit_w) + tf.nn.l2_loss(
        #     inter_w)) * FLAGS.lambda_l2
        l2_loss = tf.nn.l2_loss(out_to_logit_w) * FLAGS.lambda_l2


    logits_reshaped = tf.transpose(tf.reshape(logits,
                                              [FLAGS.sentence_length, -1,
                                               FLAGS.vocab_size]), [1, 0, 2])
    best_pred = tf.arg_max(logits_reshaped, 2)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_words[:, 1:],
        logits=logits_reshaped[:, :-1, :]) / np.log(2) * \
           tf.to_float(tf.not_equal(input_words[:, 1:],
                                    vocabulary.dict[vocabulary.PADDING]))

    # Sanity check
    any_word = input_words[10, 5]
    any_word_pre = input_words[10, 4]
    any_word_probs = tf.nn.softmax(logits_reshaped[10, 5, :])
    any_word_max_prob = tf.reduce_max(any_word_probs)
    any_word_prediction = tf.argmax(any_word_probs, dimension=0)
    any_word_real_perp = 1 / any_word_probs[any_word]

    any_word_probs2 = tf.nn.softmax(logits_reshaped[11, 6, :])
    any_word_prediction2 = tf.argmax(any_word_probs2, dimension=0)

    # output of the last layer in the unrolled LSTM
    last_output = lstm_outputs[-1]
    last_prob = tf.nn.softmax(logits_reshaped[:, -1, :])

    # add to collection for re-use in task 1.2
    tf.add_to_collection("last_output", last_output)
    tf.add_to_collection("last_prob", last_prob)
    tf.summary.histogram("last_prob", last_prob)

    # define perplexity and add to collection to provide access when reloading the model elsewhere
    # add a summary scalar for tensorboard
    # TODO Confirm that the elementwise crossentropy is -p(w_t|w_1,...,w_{t-1})
    mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(
        tf.to_float(tf.not_equal(input_words[:, 1:],
                                 vocabulary.dict[vocabulary.PADDING]))) + l2_loss
    tf.summary.scalar('loss', mean_loss)
    perplexity = tf.pow(2.0, mean_loss)
    tf.add_to_collection("perplexity", perplexity)
    tf.summary.scalar('perplexity', perplexity)



    sentence_perplexity = \
        tf.pow(2.0, tf.reduce_sum(loss, axis=1) / tf.reduce_sum(tf.to_float(
            tf.not_equal(input_words[:, 1:],
                         vocabulary.dict[vocabulary.PADDING])), axis=1))
    tf.summary.histogram('sPerplexity', sentence_perplexity)

    # TODO add learning_rate to summaries
    optimiser = tf.train.AdamOptimizer(eff_rate)

    gradients, v = zip(*optimiser.compute_gradients(mean_loss))
    # TODO from the doc it looks as if we actually wanted to set use_norm to 10 instead. Confirm!
    # https://www.tensorflow.org/api_docs/python/tf/clip_by_global_norm
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10)
    train_op = optimiser.apply_gradients(zip(clipped_gradients, v),
                                         global_step=global_counter)

    # initialize the Variables
    init_op = tf.global_variables_initializer()
    # Saver to save model checkpoints
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints,
                           keep_checkpoint_every_n_hours=2)

    # Get an idea of the overall size of the model
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        print(shape)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Built a graph with a total of %d trainable parameters" % (
        total_parameters))

    merged_summaries = tf.summary.merge_all()

    # loop over training batches
    if FLAGS.do_train:
        with tf.Session() as sess:
            # Summary Filewriter
            train_summary_dir = os.path.join(FLAGS.model_dir, "summary",
                                             "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                         sess.graph)


            # Restoring or initialising session
            if not FLAGS.force_init:
                try:
                    saver.restore(sess,
                                  tf.train.latest_checkpoint(FLAGS.model_dir))
                    print("Recovered Session")
                except:  # TODO find name for this exception (it does not accept the NotFoundError displayed if it does not find the save)
                    sess.run(init_op)
                    print("Unexpectedly initialised session")
            else:
                sess.run(init_op)
                print("Initialised session")

            #load the pretrained word embeddings from word2vec
            if FLAGS.task is not "A":
                print("assigning pretrained embedding")
                with tf.Session() as session:
                    load_embedding(session, vocabulary.get_vocabulary_as_dict(), embedding_matrix, FLAGS.embedding,
                                   FLAGS.embedding_size, FLAGS.vocab_size)

            print("Start training")
            for data_train in batches_train:
                gc_ = 0
                if (gc_ % FLAGS.evaluate_every) == 0 or gc_ == 1:
                    ms_, gc_, pp_, last_out_, last_prob_, _, \
                    word, max_p, pred, perp_of_true, word2, word_pre = \
                        sess.run([merged_summaries, global_counter, perplexity,
                                  last_output, last_prob, train_op,
                                  any_word, any_word_max_prob, any_word_prediction,
                                  any_word_real_perp, any_word_prediction2,
                                  any_word_pre],
                                 feed_dict={input_words: data_train})
                else:
                    _, gc_ = sess.run([train_op, global_counter],
                             feed_dict={input_words: data_train})

                if gc_ > FLAGS.no_output_before_n:
                    train_summary_writer.add_summary(ms_, gc_)

                if (gc_ % FLAGS.evaluate_every) == 0 or gc_ == 1:
                    print("Iteration %s: Perplexity is %s" % (gc_, pp_))

                if (gc_ % FLAGS.checkpoint_every == 0) and gc_ > 0:
                    ckpt_path = saver.save(sess, os.path.join(FLAGS.model_dir,
                                                              'model'), gc_)
                    print("Model saved in file: %s" % ckpt_path)
                if gc_ % FLAGS.hlave_lr_every == 0 & gc_ > 0:
                    eff_rate /= 2
                    print("Adjusted learning rate")
                    print(eff_rate)

                if gc_ % 250 == 0:
                    print(
                        "Target: %s, Perplexity of target: %s,  "
                        "max prob: %s, predicted: %s, second_word: %s,"
                        "Previous word: %s" % (word, perp_of_true,
                                                         max_p, pred, word2,
                                               word_pre))

    if FLAGS.do_eval:
        with tf.Session() as sess:
            # Restoring or initialising session
            saver.restore(sess,
                          tf.train.latest_checkpoint(FLAGS.model_dir))
            print("Recovered Session")
            out_pp = np.empty(0)
            for data_eval in batches_eval:
                out_pp = np.concatenate((out_pp, sess.run(sentence_perplexity,
                                                          feed_dict={
                                                              input_words: data_eval})))
            np.savetxt(
                FLAGS.output_dir + "/group25.perplexity" + FLAGS.task,
                np.array(out_pp),
                fmt="%4.8f",
                delimiter=',')

    if FLAGS.do_generate:
        with tf.Session() as sess:
            # Restoring or initialising session
            saver.restore(sess,
                          tf.train.latest_checkpoint(FLAGS.model_dir))
            print("Recovered Session")
            sentences = []
            for data_gen in batches_gen:
                input = data_gen
                for t in range(FLAGS.sentence_length - 1):
                    best = sess.run(best_pred, feed_dict={input_words: input})

                    if t < (FLAGS.sentence_length - 1):
                        next_available = data_gen[:, t+1] != vocabulary.dict[vocabulary.PADDING]
                        input[:, t + 1] = next_available * data_gen[:, t+1] + \
                                          (1-next_available) * best[:, t]

                sentences.append(input)

        translator = vocabulary.get_inverse_voc_dict()
        sentence_together = np.vstack(sentences)
        out_sentences = np.array([translator[x] for x in sentence_together.reshape([-1])]).reshape([-1, FLAGS.sentence_length])

        tt = clean_and_cut_sentences(out_sentences)

        np.savetxt(
            FLAGS.output_dir + "/group25.continuation" + FLAGS.task,
            np.array(tt),
            fmt="%s",
            delimiter='\n')



if __name__ == '__main__':
    tf.app.run()
