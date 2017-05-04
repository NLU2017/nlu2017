import numpy as np
import collections
import pickle

class SentenceCleaner:
    """ handles a single string a 'sentence' and converts it to a 1 d tensor doing all necessary preparations to it
        this class is stateless, except for global constants
    """

    LENGTH = 30


    def prepare_sentence(self, input_string, is_partial = False):
        """
        takes any input sentence (string of words separated by \" \" (SPACE) and resturns a
        numpy array which
            - has length 30
            - starts with <bos> and
            - ends with <eos>
            - contains up to 28 words from the input sentence in between
            - if the sentence is shorter the array is padded with <pad>
        """
        if(is_partial):
            words = ([Vocabulary.INIT_SEQ] + input_string.strip().split(Vocabulary.SPLIT))[:30]
        else:
            words = ([Vocabulary.INIT_SEQ] + input_string.strip().split(Vocabulary.SPLIT) + [Vocabulary.END_SEQ])[:30]
        line_array = np.full([SentenceCleaner.LENGTH], Vocabulary.PADDING, dtype='object')
        line_array[0:len(words)] = words
        return line_array




class DataLoader:
    """
    loads the trainings data and creates a generator for data batches to serve to Neural Networks
    """
    #can have this one as a class attribute since it is stateless
    cleaner = SentenceCleaner()



    def __init__(self, path, vocabulary=None, do_shuffle = True, is_partial=False):
        print("Reading data from {} ".format(path))
        self.load_data(path, vocabulary, is_partial)
        self.shuffle = do_shuffle



    def load_data(self, path, vocabulary, partial_sentence):
        """ takes the path to the data file and loads the data into memory
            data is loaded into a tensor [samples, 30]
            TODO: at the moment this loads the entire file into memory, because it is then easier to
            serve it in a shuffled mode later on, but maybe this is too much...

        """
        list = []
        with open(path) as file:
            for line in file:
                list.append(DataLoader.cleaner.prepare_sentence(line, is_partial=partial_sentence))
        self.data = np.array(list)
        print("Start translation")
        if vocabulary is not None:
            voc = vocabulary.get_vocabulary_as_dict()
            self.data_num = np.array([voc.get(w, voc["<unk>"]) for w in self.data.reshape([-1])]).reshape(self.data.shape)
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
    dict = None
    inverse_dict = None


    def load_file(self, path):
        wordcount = collections.Counter()
        with open(path) as file:
            for line in file:
                wordcount.update(line.split())
        max_words = min(Vocabulary.SIZE - len(Vocabulary.keywords), len(wordcount))
        self.words = sorted(wordcount, key=wordcount.get, reverse=True)[0:max_words]
        self.words.extend(Vocabulary.keywords)

    def get_vocabulary_as_dict(self):
        self.dict = {k:v for v, k in enumerate(self.words)}
        return self.dict

    def get_inverse_voc_dict(self):
        self.inverse_dict = {k:v for v,k in self.dict.items()}
        return self.inverse_dict

    def contains(self, word):
        """returns True if the word is one of the keywords
        or in the extracted vocabulary"""
        return word in self.words

    def is_known_keyword(self, w):
        return w in Vocabulary.keywords and w != Vocabulary.UNK

    def is_padding(self, word):
        return word == Vocabulary.PADDING

    def get_padding_key(self):
        return self.dict[Vocabulary.PADDING]

    def is_init(self, word):
        return word == Vocabulary.INIT_SEQ

    def is_init_or_pad(self, word_key):
        """returns true if the word_key (integer coding) is either the key for <pad> or <bos>"""
        if self.inverse_dict is None:
            self.get_inverse_voc_dict()
        try:
            word = self.inverse_dict[word_key]
        except KeyError:
            return False
        return self.is_init(word) or self.is_padding(word)

    def translate_to_sentence(self, list_of_keys):
        self.get_inverse_voc_dict()
        words = [self.inverse_dict[k] if not self.is_init_or_pad(k) else '' for k in
                 list_of_keys]
        result = ' '.join(words).strip()
        return result

