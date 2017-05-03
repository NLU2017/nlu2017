import sys

sys.path.append('../src')
import unittest
from utils import SentenceCleaner
from utils import DataLoader
from utils import Vocabulary
import numpy as np



class SentenceCleanerTest(unittest.TestCase):
    if __name__ == '__main__':
        unittest.main()

    def setUp(self):
        self.cleaner = SentenceCleaner()

    def test_SentenceCleaner_array_starts_with_bos(self):
        sentence = "lorem ipsum dolor sit amet , consetetur"
        append_split = self.cleaner.prepare_sentence(sentence)
        assert append_split[0] == Vocabulary.INIT_SEQ

    def test_SentenceCleaner_array_ends_with_is_eos(self):
        sentence = "lorem ipsum dolor sit amet , consetetur"
        append_split = self.cleaner.prepare_sentence(sentence)
        assert append_split[8] == Vocabulary.END_SEQ

    def test_SentenceCleaner_returns_array_with_shape_30(self):
        sentence = "Aliquam ut mattis felis , vel accumsan orci . Donec arcu dolor , luctus in nibh id , rutrum consequat elit . Phasellus sed dolor maximus , euismod quam sed , varius tellus . Fusce condimentum libero in ante lobortis"
        append_split = self.cleaner.prepare_sentence(sentence)
        assert append_split.shape[0] == SentenceCleaner.LENGTH

    def test_array_is_padded(self):
        sentence = "lorem ipsum dolor sit amet , consetetur"
        ar = self.cleaner.prepare_sentence(sentence)
        assert ar[8] == Vocabulary.END_SEQ
        for i in range(9, 29):
            assert ar[i] == Vocabulary.PADDING

    def test_array_is_padded_for_partial_sentence(self):
        sentence = "lorem ipsum dolor sit amet , consetetur"
        ar = self.cleaner.prepare_sentence(sentence, is_partial=True)
        assert ar[8] == Vocabulary.PADDING
        for i in range(9, 29):
            assert ar[i] == Vocabulary.PADDING



class DataLoaderTest(unittest.TestCase):
    if __name__ == '__main__':
        unittest.main()

    def setUp(self):
        self.voc = Vocabulary()
        self.voc.load_file("./test_sentences.txt")
        self.loader = DataLoader("./test_sentences.txt", self.voc, do_shuffle=False)

    def test_loadFileIntoMemory(self):
        # loading data is done in constructor
        assert self.loader.data_num is not None
        assert self.loader.data_num.shape == (6, 30)


    def test_iterate_over_all_epochs_and_batches(self):
        batches = self.loader.batch_iterator(3, 3)
        count = 0

        for i in batches:
            count += 1
            assert i.shape == (3, 30)
        assert count == 6

    def test_each_sentence_has_bos_eos(self):
        assert np.sum(np.equal(self.loader.data, Vocabulary.END_SEQ)) == self.loader.data.shape[0]
        assert np.sum(np.equal(self.loader.data, Vocabulary.INIT_SEQ)) == self.loader.data.shape[0]

    def test_load_partial_sentence_no_eos(self):
        self.loader = DataLoader("./test_sentences.txt", self.voc, do_shuffle=False, is_partial=True)
        assert np.sum(np.equal(self.loader.data, Vocabulary.END_SEQ)) == 0
        assert np.sum(np.equal(self.loader.data, Vocabulary.INIT_SEQ)) == self.loader.data.shape[0]


class VocabularyTest(unittest.TestCase):
    def setUp(self):
        self.voc = Vocabulary()
        Vocabulary.SIZE = 7


    def test_load_file(self):
        self.voc.load_file("./test_vocabulary.txt")
        assert self.voc.contains("zwei")
        assert self.voc.contains("drei")
        assert self.voc.contains("eins")
        assert self.voc.contains(Vocabulary.UNK)
        assert self.voc.contains(Vocabulary.END_SEQ)
        assert self.voc.contains(Vocabulary.INIT_SEQ)
        assert self.voc.contains(Vocabulary.PADDING)

        assert not self.voc.contains("vier")
        assert not self.voc.contains("sieben")
        assert not self.voc.contains("sechs")


    def test_get_vocabulary_as_dict(self):
        self.voc.load_file("./test_vocabulary.txt")
        dict = self.voc.get_vocabulary_as_dict()

        for k, val in enumerate(Vocabulary.keywords):
            assert dict[val] == 3 + k
        assert dict["drei"] == 0
        assert dict["zwei"] == 1
        assert dict["eins"] == 2

    def test_invert_vocabulary_dict(self):
        self.voc.load_file("./test_vocabulary.txt")
        dict = self.voc.get_vocabulary_as_dict()
        inverted = self.voc.get_inverse_voc_dict()
        print(dict)
        print(inverted)
        for key, val in dict.items():
            assert inverted[val] == key

    def test_is_keyword(self):
        self.voc.load_file("./test_vocabulary.txt")
        assert not self.voc.is_known_keyword("drei")
        assert not self.voc.is_known_keyword("<unk>")
        assert self.voc.is_known_keyword("<eos>")
        assert self.voc.is_known_keyword("<bos>")
        assert self.voc.is_known_keyword("<pad>")

    def test_is_padding(self):
        self.voc.load_file("./test_vocabulary.txt")
        assert not self.voc.is_padding("drei")
        assert not self.voc.is_padding("zwei")
        assert not self.voc.is_padding("foo")

        assert not self.voc.is_padding("<unk>")
        assert not self.voc.is_padding("<eos>")
        assert not self.voc.is_padding("<bos>")
        assert self.voc.is_padding("<pad>")

    def test_is_padding(self):
        self.voc.load_file("./test_vocabulary.txt")
        assert not self.voc.is_init("drei")
        assert not self.voc.is_init("zwei")
        assert not self.voc.is_init("foo")

        assert not self.voc.is_init("<unk>")
        assert not self.voc.is_init("<eos>")
        assert  self.voc.is_init("<bos>")
        assert not self.voc.is_init("<pad>")

    def test_is_pad_or_init(self):
        self.voc.load_file("./test_vocabulary.txt")
        dict = self.voc.get_vocabulary_as_dict()
        assert not self.voc.is_init_or_pad(dict["drei"])
        assert not self.voc.is_init_or_pad(dict["zwei"])
        assert not self.voc.is_init_or_pad(50000)

        assert not self.voc.is_init_or_pad(dict["<unk>"])
        assert not self.voc.is_init_or_pad(dict["<eos>"])
        assert self.voc.is_init_or_pad(dict["<bos>"])
        assert self.voc.is_init_or_pad(dict["<pad>"])


