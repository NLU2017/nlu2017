import time
import os
import tensorflow as tf
from utils import DataLoader
from utils import Vocabulary


## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("train_file_path", "../data/sentences.train", "Path to the training data")
# TODO: Model parameters

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("log_dir", "../runs/", "Output directory (default: '../runs/')")
# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

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

    #load training data
    train_loader = DataLoader(FLAGS.train_file_path, vocabulary, do_shuffle=True)
    batches_train = train_loader.batch_iterator(FLAGS.num_epochs, FLAGS.batch_size)

    #loop over training batches
    for data_train in batches_train:
        pass



if __name__ == '__main__':
    tf.app.run()
