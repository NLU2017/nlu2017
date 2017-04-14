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
        words = input_string.split(self.SPLIT)
        t = [words[i] if i < len(words) else self.PADDING for i in range(30)]
        line_array = np.ndarray([30], dtype='object')
        line_array[0] = self.INIT_SEQ
        line_array[1:self.LENGTH] = t[0:self.LENGTH-1]
        line_array[-1] = self.END_SEQ
        return line_array




class DataLoader:

    cleaner = SentenceCleaner()



    def load_data(self, path):
        """ takes the path to the data file and loads the data into memory
            data is loaded into a tensor [samples, 30]
            TODO: at the moment this loads the entire file into memory, because it is then easier to
            serve it in a shuffled mode later on, but maybe this is too much...

        """
        list = []
        with open(path) as file:
            for line in file:
                list.append(self.cleaner.prepare_sentence(line))
        self.data = np.array(list)




