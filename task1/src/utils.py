import numpy as np

class SentenceCleaner:
    LENGTH = 30
    PADDING = "<pad>"
    INIT_SEQ = "<bos>"
    END_SEQ = "<eos>"
    SPLIT = " "

    def prepare_sentence(self, input_string):
        line_array = np.ndarray([30], dtype='object')
        line_array[0] = self.INIT_SEQ
        word_list = input_string.split(self.SPLIT)
        end = min(len(word_list), self.LENGTH-1)
        line_array[1:end + 1] = word_list[0:end]
        line_array[-1] = self.END_SEQ

        return line_array



