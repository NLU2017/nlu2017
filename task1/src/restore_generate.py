import time
import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary
import numpy as np
import csv


tf.flags.DEFINE_string("cont_file_path", "../data/sentences.mycontinuation",
                       "Path to the continuation data (default ../data/sentences.continuation)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train",
                       "Path to the training data")
tf.flags.DEFINE_integer("sentence_length", 20, "Length of the input sentences (default: 20)")
tf.flags.DEFINE_string("log_dir", "../runs/1493149871", "Checkpoint directory")
tf.flags.DEFINE_string("meta_graph_file", "model-200.meta", "Name of meta graph file")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")

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

        #TODO: which op do we really need??'
        last_output = tf.get_collection("last_output")
        input_words = tf.get_collection("input_words")



        input_words = tf.get_collection('input_words')[0]




        def generate_sentence(sess, start_data):
            sentences = []
            batches = start_data.batch_iterator(1, FLAGS.batch_size)
            for b in batches:
                input = b
                new_sentence = np.ndarray([FLAGS.batch_size, FLAGS.sentence_length ], dtype='object')
                new_sentence[:, 0] = b[:,0]
                for t in range(FLAGS.sentence_length):


                    logits = sess.run([last_output], feed_dict= {input_words: new_sentence})
                    # get argmax(logits) or sample(logits)
                    best_match = tf.arg_max(logits)
                    for i in range(b.shape[0]):
                        new_sentence[i, t+ 1] = if
                    has_word_in_sentence = not np.equals(b[:, t+1], Vocabulary.PADDING)
                    for i in

                    new_sentence[:, t+1] =
            print(new_sentence)
            sentences.append(new_sentence)

        generate_sentence(sess, start_data)


if __name__ == '__main__':
    tf.app.run()







