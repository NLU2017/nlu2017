import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary, SentenceCleaner
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file



tf.flags.DEFINE_string("cont_file_path", "../data/sentences.continuation_short",
                       "Path to the continuation data (default ../data/sentences.continuation)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_integer("sentence_length", 20, "Length of the input sentences (default: 20)")
tf.flags.DEFINE_string("log_dir", "../runs/1493922093", "Checkpoint directory (f.ex. ../runs/1493459028")
tf.flags.DEFINE_string("check_point","model-200.meta", "Name of meta graph file (f.ex.  model-400.meta)")
tf.flags.DEFINE_integer("batch_size", 1, "Batch size (default: 64)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def main(unused_argv):
    #load the continuation file
    vocabulary = Vocabulary()
    vocabulary.load_file(FLAGS.train_file_path)
    vocabulary.get_vocabulary_as_dict()
    vocabulary.get_inverse_voc_dict()

    # load training data
    start_data = DataLoader(FLAGS.cont_file_path,
                              vocabulary, do_shuffle=False, is_partial=True)

    #print_tensors_in_checkpoint_file(os.path.join(FLAGS.log_dir, 'model-200'), tensor_name='', all_tensors=True)


    with tf.Session() as sess:


        restore_from = os.path.join(FLAGS.log_dir, FLAGS.check_point)
        ckpt = tf.train.import_meta_graph(restore_from)
        # Restore variables.
        ckpt.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))


        ## helper function: prints all tensors saved in the file
        #print_tensors_in_checkpoint_file(os.path.join(FLAGS.log_dir, 'model-200'), tensor_name='', all_tensors=True)
        num_steps = 1
        input_words = tf.placeholder(tf.int32, [None, num_steps])

        probabilities = create_model(input_words, time_steps=num_steps)

        def generate_sentence(sess, start_data):

            batches = start_data.batch_iterator(1, FLAGS.batch_size)
            sentences = []
            for b in batches:
                # there must be at least one word in the input sentence because the all start with <bos>
                sentence_input = np.ndarray((b.shape[0], FLAGS.sentence_length + 1), dtype=np.int32)
                sentence_input[:, 0:1] = b[:,0:1]
                for t in range(0, FLAGS.sentence_length):
                    #print("input {}".format(sentence_input[:,t:t+1]))
                    feed = {input_words: sentence_input[:,t:t+1]}
                    logits= sess.run([probabilities], feed_dict= feed)
                    # get argmax(logits) along dimension 1 (= size of dictionary)
                    best_match = np.argmax(logits[0], axis=1)
                    #print_word("best_match", best_match, vocabulary)

                    next_word_in_input = b[:, t+1]
                    #print_word("next_from_input", next_word_in_input, vocabulary)
                    has_word_in_sentence = next_word_in_input != vocabulary.get_padding_key()
                    #add the next word of the input sentence for the next run or the best_fit from the model if the sentence input is exhausted
                    # if the sentence input is exhausted
                    next_words = np.asarray([next_word_in_input[s] if has_word_in_sentence[s] else
                                             best_match[s] for s in range(b.shape[0])])
                    sentence_input[:, t+1] = next_words


                #translate back to Strings and store
                #TODO how to deal with unknown words?
                #at the moment we replace them by <unk> in the input data. should they be replace to to what they were
                #or can we keep the <unk>
                for sample in range(b.shape[0]):
                    result = vocabulary.translate_to_sentence(sentence_input[sample, :])
                    sentences.append(result)
                    print(result)
            return sentences

        #run it
        sess.run(tf.global_variables_initializer())
        sentences = []
        sentences.append(generate_sentence(sess, start_data))
        np.savetxt(
            os.path.join(FLAGS.log_dir + "/group25.continuation"),
            np.asarray(sentences),
            fmt="%s",
            delimiter='\n')


def print_word(text, index_vector, vocabulary):

    for i in range(index_vector.shape[0]):
        print("{} {}: {} = {}".format(text, i, index_vector[i], vocabulary.inverse_dict[index_vector[i]]))


def create_model(input_words, time_steps=1, vocab_size=20000, embedding_size = 100, lstm_size = 512):

    W = tf.get_variable(name="output_weights", shape=[lstm_size, vocab_size])
    b = tf.get_variable(name="output_bias", shape=[vocab_size])

    embedding_matrix = tf.get_variable('embedding_matrix', shape=[vocab_size, embedding_size])

    # Create the graph
    embedded_words = tf.nn.embedding_lookup(embedding_matrix, input_words)
    # RNN graph
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm_zero_c = tf.get_variable("zero_state_c", shape=[1, lstm_size])
    lstm_zero_h = tf.get_variable("zero_state_h", shape=[1, lstm_size])
    lstm_state = (tf.tile(lstm_zero_c, [tf.shape(input_words)[0], 1]),
                  tf.tile(lstm_zero_h, [tf.shape(input_words)[0], 1]))

    outputs = []
    with tf.variable_scope("RNN"):
        for ts in range(time_steps):
            if ts > 0:
                tf.get_variable_scope().reuse_variables()
            lstm_out, lstm_state = lstm_cell(embedded_words[:, ts, :], lstm_state)
            outputs.append(lstm_out)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1,lstm_size ])
    logits = tf.matmul(output, W) + b
    probabilities = tf.nn.softmax(logits)
    return probabilities


if __name__ == '__main__':
    tf.app.run()







