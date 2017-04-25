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
tf.flags.DEFINE_string("log_dir", None, "Checkpoint directory")
tf.flags.DEFINE_string("meta_graph_file", None, "Name of meta graph file")
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

    # load training data
    start_data = DataLoader(FLAGS.cont_file_path,
                              vocabulary, do_shuffle=False)
    batches = start_data.batch_iterator(FLAGS.num_epochs, FLAGS.batch_size)

    with tf.Session() as sess:
        # Restore computation graph.
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        # Restore variables.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))




