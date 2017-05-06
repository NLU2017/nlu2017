import time
import os
import tensorflow as tf
from utils import Vocabulary, DataLoader
import numpy as np
import csv
import pickle as pickle

## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_string("cont_path", "../data/sentences.continuation_short", "path to sentence continuation file")
tf.flags.DEFINE_string("eval_file_path", "../data/sentences.eval",
                       "Path to the validation data")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the input sentences")
tf.flags.DEFINE_integer("vocab_size", 20000,
                        "Number of words in the vocabulary")

tf.flags.DEFINE_string("embedding", "../data/wordembeddings-dim100.word2vec",
                       "Path to the embedding file (space separated)")

# Model parameters
tf.flags.DEFINE_integer("lstm_size", 512, "Length of the hidden state")
tf.flags.DEFINE_integer("embedding_size", 100, "Dimension of the embedding")
tf.flags.DEFINE_string("task", "C", "Task to be solved")
tf.flags.DEFINE_integer("intermediate_size", 512,
                        "Dimension of down-projection in task C")
tf.flags.DEFINE_string("model_name", str(int(time.time())), "Name the model")
tf.flags.DEFINE_bool("is_continuation", False, "run only prediction for task 1.2" )

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3,
                        "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("log_dir", "../runs/",
                       "Output directory (default: '../runs/')")
tf.flags.DEFINE_float("learning_rate", 1e-2,
                      "Inital learning rate of the optimizer")
tf.flags.DEFINE_integer("hlave_lr_every", 10000,
                        "Every n steps the learning rate is halved")
tf.flags.DEFINE_float("dropout_rate", 0.0,
                      "Dropout probs. (0.0 for no dropout)")
tf.flags.DEFINE_integer("no_output_before_n", 5000,
                        "Supress the first outputs, because of strong changes")
tf.flags.DEFINE_boolean("allow_batchnorm", False,
                        "Allow or disallow batch normalisation")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")
tf.flags.DEFINE_boolean("force_init", True,
                        "Whether to always start training from scratch")
tf.flags.DEFINE_boolean("is_training", False, "training mode")

tf.flags.DEFINE_boolean("is_eval", False, "do evaluation")
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
    vocabulary = pickle.load( open( "./vocabulary.pickle", "rb" ))
    # load training data
    if FLAGS.is_continuation:
        cont_loader = DataLoader(FLAGS.cont_path, vocabulary, do_shuffle=False, is_partial=True)
        batches_cont = cont_loader.batch_iterator(1, FLAGS.batch_size)
    if FLAGS.is_training:
        train_loader = DataLoader(FLAGS.train_file_path,
                              vocabulary, do_shuffle=True)
        batches_train = train_loader.batch_iterator(FLAGS.num_epochs,
                                                FLAGS.batch_size)
    if FLAGS.is_eval:
        # load validation data

        eval_loader = DataLoader(FLAGS.eval_file_path,
                             vocabulary, do_shuffle=False)
        batches_eval = eval_loader.batch_iterator(num_epochs=1, batch_size=FLAGS.batch_size)

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
                print("Unmatched: %s" % k)
                external_embedding[v, :] = \
                    np.random.uniform(low=-0.25,
                                      high=0.25,
                                      size=FLAGS.embedding_size)
        embedding_matrix = tf.Variable(external_embedding, dtype=tf.float32)

    is_training = tf.placeholder(tf.bool)
    tf.add_to_collection("is_training", is_training)
    input_words = tf.placeholder(tf.int32, [None, FLAGS.sentence_length])
    # add to collection for usage from restored model
    tf.add_to_collection("input_words", input_words)

    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)


    embedded_words_bn = \
        tf.layers.batch_normalization(embedded_words,
                                      training=is_training,
                                      center=FLAGS.allow_batchnorm,
                                      scale=FLAGS.allow_batchnorm)


    tf.add_to_collection("embedded_words", embedded_words_bn)

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

    if not FLAGS.task == "C":
        out_to_logit_w = tf.get_variable(
            name="output_weights",
            shape=[FLAGS.lstm_size, FLAGS.vocab_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        out_to_logit_b = tf.get_variable("output_bias", shape=[FLAGS.vocab_size])

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
        out_to_logit_b = tf.get_variable("output_bias", shape = [FLAGS.vocab_size])

    # initialize

    lstm_outputs = []
# add summaries for tensorboard

    with tf.variable_scope("RNN"):


        sentences = []
        best_match = [input_words[:,0]]
        for time_step in range(FLAGS.sentence_length):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            if FLAGS.is_continuation:

                has_word_in_sequence = tf.to_int32(tf.not_equal(input_words[:, time_step], vocabulary.dict[vocabulary.PADDING]))
                is_padding = tf.to_int32(tf.equal(input_words[:, time_step], vocabulary.dict[vocabulary.PADDING]))

                # add the next word of the input sentence for the next run or the best_fit from the model if the sentence input is exhausted
                # if the sentence input is exhausted
                next_words = best_match * is_padding + input_words[:, time_step] * has_word_in_sequence
                sentences.append(next_words[0])
                loop_input = tf.nn.embedding_lookup(embedding_matrix, next_words[0])
            else:
                loop_input = embedded_words_bn[:, time_step, :]
                sentences.append(input_words[:, time_step])

            lstm_out, lstm_state = lstm(loop_input, lstm_state)
            lstm_outputs.append(lstm_out)

            if FLAGS.is_continuation:

                logits_p = tf.matmul(tf.matmul(lstm_out, inter_w) + inter_b,
                                     out_to_logit_w) + out_to_logit_b
                #logits_p = tf.matmul(lstm_out, out_to_logit_w) + out_to_logit_b
                best_match = tf.cast(tf.arg_max(logits_p, dimension=1), tf.int32)

                # has_word_in_sequence =tf.to_int32(tf.not_equal(input_words[:,time_step +1], vocabulary.dict[vocabulary.PADDING]))
                # is_padding =tf.to_int32( tf.equal(input_words[:,time_step +1], vocabulary.dict[vocabulary.PADDING]))
                #
                # # add the next word of the input sentence for the next run or the best_fit from the model if the sentence input is exhausted
                # # if the sentence input is exhausted
                #
                # next_words = best_match * is_padding + input_words[:, time_step + 1] * has_word_in_sequence
                #
                #
                # loop_input = tf.nn.embedding_lookup(embedding_matrix, next_words)

                sentences.append(next_words)
            # else:
            #     loop_input = embedded_words_bn[:, time_step + 1, :]
            #     sentences.append(input_words[:, time_step + 1])



    output = tf.concat(axis=0, values=lstm_outputs)

    lstm_out_drop = tf.layers.dropout(output,
                                      rate=FLAGS.dropout_rate,
                                      training=is_training)

    if not FLAGS.task == "C":
        logits = tf.matmul(lstm_out_drop, out_to_logit_w) + out_to_logit_b
    else:
        logits = tf.matmul(tf.matmul(lstm_out_drop, inter_w) + inter_b,
                           out_to_logit_w) + out_to_logit_b


    logits_reshaped = tf.transpose(tf.reshape(logits,
                                              [FLAGS.sentence_length, -1,
                                               FLAGS.vocab_size]), [1, 0, 2])



    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_words[:, 1:],
        logits=logits_reshaped[:, :-1, :]) / np.log(2) * \
           tf.to_float(tf.not_equal(input_words[:, 1:],
                                    vocabulary.dict[vocabulary.PADDING]))


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
                                 vocabulary.dict[vocabulary.PADDING])))
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
    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate)

    gradients, v = zip(*optimiser.compute_gradients(loss))
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
    with tf.Session() as sess:
        # Summary Filewriter
        train_summary_dir = os.path.join(FLAGS.model_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)

        # Restoring or initialising session
        if not FLAGS.force_init or not FLAGS.is_continuation or not FLAGS.is_eval:
            try:
                print(FLAGS.model_dir)
                saver.restore(sess,
                              tf.train.latest_checkpoint(FLAGS.model_dir))
                print("Recovered Session")
            except:  # TODO find name for this exception (it does not accept the NotFoundError displayed if it does not find the save)
                sess.run(init_op)
                print("Unexpectedly initialised session")
        else:
            sess.run(init_op)
            print("Initialised session")

        if FLAGS.is_training:
            print("Start training")
            for data_train in batches_train:
                ms_, gc_, pp_, last_out_, last_prob_, _, = \
                sess.run([merged_summaries, global_counter, perplexity,
                          last_output, last_prob, train_op],
                         feed_dict={input_words: data_train,
                                    is_training: True})

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

        if FLAGS.is_eval:
            print("Start validation")
            out_pp = np.empty(0)
            for data_eval in batches_eval:
                out_pp = np.concatenate((out_pp, sess.run(sentence_perplexity,
                                                      feed_dict={
                                                          input_words: data_eval,
                                                          is_training: False})))
            np.savetxt(
                FLAGS.output_dir + "/group25.perplexity" + FLAGS.task,
                np.array(out_pp),
                fmt="%4.8f",
                delimiter=',')

        if FLAGS.is_continuation:
            output_indices = []
            for data_eval in batches_cont:
                set = sess.run(sentences, feed_dict={input_words: data_eval, is_training:False})
                output_indices.append(set)

            sent = np.asarray(output_indices)[0]
            sent = np.transpose(sent)

            print(sent.shape)

            for i in range(sent.shape[0]):
                w = vocabulary.translate_to_sentence(sent[i])
                print(sent[i])
                print(w)

if __name__ == '__main__':
    tf.app.run()
