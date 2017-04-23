import numpy as np
import collections
import pickle

class SentenceCleaner:
    """ handles a single string a 'sentence' and converts it to a 1 d tensor doing all necessary preparations to it
        this class is stateless, except for global constants
    """

    LENGTH = 30


    def prepare_sentence(self, input_string):
        """
        takes any input sentence (string of words separated by \" \" (SPACE) and resturns a
        numpy array which
            - has length 30
            - starts with <bos> and
            - ends with <eos>
            - contains up to 28 words from the input sentence in between
            - if the sentence is shorter the array is padded with <pad>
        """
        words = input_string.strip().split(Vocabulary.SPLIT)
        t = [words[i] if i < len(words) else Vocabulary.PADDING for i in range(SentenceCleaner.LENGTH)]
        line_array = np.ndarray([SentenceCleaner.LENGTH], dtype='object')
        line_array[0] = Vocabulary.INIT_SEQ
        line_array[1:SentenceCleaner.LENGTH] = t[0:SentenceCleaner.LENGTH-1]
        line_array[-1] = Vocabulary.END_SEQ
        return line_array




class DataLoader:
    """
    loads the trainings data and creates a generator for data batches to serve to Neural Networks
    """
    #can have this one as a class attribute since it is stateless
    cleaner = SentenceCleaner()



    def __init__(self, path, store_path, vocabulary=None, do_shuffle = True):
        print("Reading data from {} ".format(path))
        self.load_data(path, vocabulary, store_path)
        self.shuffle = do_shuffle



    def load_data(self, path, vocabulary, store_path):
        """ takes the path to the data file and loads the data into memory
            data is loaded into a tensor [samples, 30]
            TODO: at the moment this loads the entire file into memory, because it is then easier to
            serve it in a shuffled mode later on, but maybe this is too much...

        """
        list = []
        with open(path) as file:
            for line in file:
                list.append(DataLoader.cleaner.prepare_sentence(line))
        self.data = np.array(list)
        print("Start translation")
        if vocabulary is not None:
            voc = vocabulary.get_vocabulary_as_dict()
            self.data_num = np.array([voc.get(w, voc["<unk>"]) for w in self.data.reshape([-1])]).reshape(self.data.shape)

        pickle.dump(self, open(store_path + "data_clean.pkl", "wb"))
        # TODO decide whether we still need self.data hereafter. If not, drop.

    def batch_iterator(self, num_epochs, batch_size):
        """
        creates a generator of data batches to be passed into one training run
        :param num_epochs: total sweeps over all data
        :param batch_size: nr of samples in one data batch
        :return: batch of size batch_size
        """
        num_samples = self.data_num.shape[0]
        batches_per_epoch = int((num_samples - 1) / batch_size) + 1
        for epoch in range(num_epochs):

            # Shuffle the data at each epoch
            if self.shuffle:
                shuffle_indices = np.random.permutation(np.arange(num_samples))
                shuffled_data = self.data_num[shuffle_indices]
            else:
                shuffled_data = self.data_num
            for b in range(batches_per_epoch):
                start_index = b * batch_size
                end_index = min((b + 1) * batch_size, num_samples)
                yield shuffled_data[start_index:end_index]



class Vocabulary:
    SIZE = 20000

    PADDING = "<pad>"
    INIT_SEQ = "<bos>"
    END_SEQ = "<eos>"
    UNK = "<unk>"
    SPLIT = " "
    keywords = [PADDING, END_SEQ, INIT_SEQ, UNK]

    def load_file(self, path):
        wordcount = collections.Counter()
        with open(path) as file:
            for line in file:
                wordcount.update(line.split())
        max_words = min(Vocabulary.SIZE - len(Vocabulary.keywords), len(wordcount))
        self.words = sorted(wordcount, key=wordcount.get, reverse=True)[0:max_words]
        self.words.extend(Vocabulary.keywords)

    def get_vocabulary_as_dict(self):
        return {k:v for v, k in enumerate(self.words)}

    def contains(self, word):
        """returns True if the word is one of the keywords
        or in the extracted vocabulary"""
        return word in self.words

