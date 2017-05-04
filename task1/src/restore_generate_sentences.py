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
tf.flags.DEFINE_string("log_dir", None, "Checkpoint directory (f.ex. ../runs/1493459028")
tf.flags.DEFINE_string("meta_graph_file",None, "Name of meta graph file (f.ex.  model-400.meta)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size (default: 64)")

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
    vocabulary.get_inverse_voc_dict()

    # load training data
    start_data = DataLoader(FLAGS.cont_file_path,
                              vocabulary, do_shuffle=False, is_partial=True)


    with tf.Session() as sess:
        # Restore computation graph.
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        # Restore variables.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
        print_tensors_in_checkpoint_file(os.path.join(FLAGS.log_dir, 'model-200'), tensor_name='', all_tensors=True)

        last_prob = tf.get_collection('last_prob')[0]
        input_words = tf.get_collection('input_words')[0]
        is_training = tf.get_collection('is_training')[0]
        last_state = tf.get_collection('last_state')[0]

        #create a graph with 1 state


        def generate_sentence(sess, start_data, init_state):

            batches = start_data.batch_iterator(1, FLAGS.batch_size)
            sentences = []
            for b in batches:
                sentence_input = np.ndarray((b.shape[0], SentenceCleaner.LENGTH), dtype=np.int32)
                sentence_input.fill(voc_dict[Vocabulary.PADDING])
                # there must be at least one word in the input sentence because the all start with <bos>
                sentence_input[:, 0:2] = b[:,0:2]
                for t in range(1, FLAGS.sentence_length):
                    logits = sess.run(last_prob, feed_dict= {input_words: sentence_input, is_training:False})
                    # get argmax(logits) along dimension 1 (= size of dictionary)
                    best_match = np.argmax(logits, axis=1)
                    #print_word("best_match", best_match, vocabulary)

                    next_word_in_input = b[:, t+1]
                    #print_word("next_from_input", next_word_in_input, vocabulary)
                    has_word_in_sentence = next_word_in_input != voc_dict[Vocabulary.PADDING]
                    #add the next word of the input sentence for the next run or the best_fit from the model if the sentence input is exhausted
                    # if the sentence input is exhausted
                    next_words = np.asarray([next_word_in_input[s] if has_word_in_sentence[s] else
                                             best_match[s] for s in range(b.shape[0])])
                    #print_word("next_words", next_words, vocabulary)
                    sentence_input[:, t + 1] = next_words
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


if __name__ == '__main__':
    tf.app.run()







