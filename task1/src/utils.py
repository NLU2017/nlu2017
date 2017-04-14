import numpy as np

class SentenceCleaner:
    LENGTH = 30
    PADDING = "<pad>"
    INIT_SEQ = "<bos>"
    END_SEQ = "<eos>"
    SPLIT = " "

    def prepare_sentence(self, input_string):
        words = input_string.split(self.SPLIT)
        t = [words[i] if i < len(words) else self.PADDING for i in range(30)]
        line_array = np.ndarray([30], dtype='object')
        line_array[0] = self.INIT_SEQ
        line_array[1:self.LENGTH] = t[0:self.LENGTH-1]
        line_array[-1] = self.END_SEQ
        return line_array



