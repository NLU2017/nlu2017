import numpy as np

class SentenceCleaner:
    """ handles a single string a 'sentence' and converts it to a 1 d tensor doing all necessary preparations to it
        this class is stateless, except for global constants
    """

    LENGTH = 30
    PADDING = "<pad>"
    INIT_SEQ = "<bos>"
    END_SEQ = "<eos>"
    SPLIT = " "


    def prepare_sentence(self, input_string):
        """
        takes any input sentencse (string of words separated by \" \" (SPACE) and resturns a
        numpy array which
            - has length 30
            - starts with <bos> and
            - ends with <eos>
            - contains up to 28 words from the input sentence in between
            - if the sentence is shorter the array is padded with <pad>
        """
        words = input_string.strip().split(SentenceCleaner.SPLIT)
        t = [words[i] if i < len(words) else SentenceCleaner.PADDING for i in range(30)]
        line_array = np.ndarray([30], dtype='object')
        line_array[0] = SentenceCleaner.INIT_SEQ
        line_array[1:SentenceCleaner.LENGTH] = t[0:SentenceCleaner.LENGTH-1]
        line_array[-1] = SentenceCleaner.END_SEQ
        return line_array




class DataLoader:
    """
    loads the trainings data and creates a generator for data batches to serve to Neural Networks
    """

    UNKNOWN = "<unk>"


    #can have this one as a class attribute since it is stateless
    cleaner = SentenceCleaner()



    def __init__(self, path, do_shuffle = True):
        print("Reading data from {} ".format(path))
        self.load_data(path)
        self.shuffle = do_shuffle


    def load_data(self, path):
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
        vocabulary = Vocabulary()
        vocabulary.extract(self.data)
        dim = self.data.shape
        for x in dim[0]:
            for y in dim[1]:
                if not vocabulary.contains(self.data[x,y]):
                    self.data[x][y] == DataLoader.UNKNOWN


    def batch_iterator(self, num_epochs, batch_size):
        """
        creates a generator of data batches to be passed into one training run
        :param num_epochs: total sweeps over all data
        :param batch_size: nr of samples in one data batch
        :return: batch of size batch_size
        """
        num_samples = self.data.shape[0]
        batches_per_epoch = int((num_samples - 1) / batch_size) + 1
        for epoch in range(num_epochs):

            # Shuffle the data at each epoch
            if self.shuffle:
                shuffle_indices = np.random.permutation(np.arange(num_samples))
                shuffled_data = self.data[shuffle_indices]
            else:
                shuffled_data = self.data
            for b in range(batches_per_epoch):
                start_index = b * batch_size
                end_index = min((b + 1) * batch_size, num_samples)
                yield shuffled_data[start_index:end_index]





class Vocabulary:
    SIZE = 20000


    def extract(self, data):
        unic, cts = np.unique(data, return_counts=True)
        unic = unic[np.argsort(-cts)]
        max_words = min(unic.shape[0], Vocabulary.SIZE)
        self.words = unic[0:max_words]

    def contains(self, word):
        return word in self.words
