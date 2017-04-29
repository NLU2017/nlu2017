import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary, SentenceCleaner
import numpy as np
import csv



tf.flags.DEFINE_string("cont_file_path", "../data/sentences.continuation",
                       "Path to the continuation data (default ../data/sentences.continuation)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_integer("sentence_length", 20, "Length of the input sentences (default: 20)")
tf.flags.DEFINE_string("log_dir", "../runs/1493459028", "Checkpoint directory")
tf.flags.DEFINE_string("meta_graph_file", "model-400.meta", "Name of meta graph file")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size (default: 32)")

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
    voc_dict = vocabulary.get_vocabulary_as_dict()

    # load training data
    start_data = DataLoader(FLAGS.cont_file_path,
                              vocabulary, do_shuffle=False)

    with tf.Session() as sess:
        # Restore computation graph.
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        # Restore variables.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

        last_prob = tf.get_collection("last_prob")[0]
        input_words = tf.get_collection('input_words')[0]

        def generate_sentence(sess, start_data):

            batches = start_data.batch_iterator(1, FLAGS.batch_size)
            sentences = []
            for b in batches:
                sentence_input = np.ndarray((b.shape[0], SentenceCleaner.LENGTH), dtype=np.int32)
                sentence_input.fill(voc_dict[Vocabulary.PADDING])
                # there must be at least one word in the input sentence because the all start with <bos>
                sentence_input[:, 0] = b[:,0]
                for t in range(FLAGS.sentence_length):
                    #print("running with input: {}".format(sentence_input))
                    logits = sess.run([last_prob], feed_dict= {input_words: sentence_input})
                    # get argmax(logits) along dimension 1 (= size of dictionary)
                    best_match = np.argmax(logits[0], axis=1)
                    next_word_in_input = b[:, t+1]
                    has_word_in_sentence = next_word_in_input != voc_dict[Vocabulary.PADDING]
                    #add the next word of the input sentence for the next run or the best_fit from the model if the sentence input is exhausted
                    next_words = np.asarray(
                        [next_word_in_input[s] if has_word_in_sentence[s] else                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   best_match[s] for s in range(b.shape[0])])
                    sentence_input[:, t + 1] = next_words
                #translate back to Strings and store
                #TODO how to deal with unknown words?
                for sample in range(b.shape[0]):
                    result = vocabulary.translate_to_sentence(sentence_input[sample, :])
                    sentences.append(result)
                    print(result)
            return sentences

        #run it
        sentences = []
        sentences.append(generate_sentence(sess, start_data))
        #TODO: print to file..
        np.savetxt(
            os.path.join(FLAGS.log_dir + "/group25.continuation"),
            np.asarray(sentences),
            fmt="%s",
            delimiter='\n')


if __name__ == '__main__':
    tf.app.run()







