import numpy as np

class SentenceCleaner:
    LENGTH = 30
    PADDING = "<pad>"
    INIT_SEQ = "<bos>"
    END_SEQ = "<eos>"
    SPLIT = " "

    def prepare(self, input_string):
        output = np.ndarray([1, 30], dtype='object')
        output[0] = self.INIT_SEQ
        word_list = input_string.split(self.SPLIT)
        end = np.min(len(word_list, self.LENGTH))
        output[1:end] = word_list[0:end]
        output[-1] = self.END_SEQ
        word_list.insert(0, self.INIT_SEQ)

        return word_list



def load_data(path):
    with open(path) as file:
        for line in file:
            res = prepare_sentence(line)


def prepare_sentence(line):
    return line