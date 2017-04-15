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
        assert append_split[-1] == Vocabulary.END_SEQ

    def test_SentenceCleaner_returns_array_with_shape_30(self):
        sentence = "Aliquam ut mattis felis , vel accumsan orci . Donec arcu dolor , luctus in nibh id , rutrum consequat elit . Phasellus sed dolor maximus , euismod quam sed , varius tellus . Fusce condimentum libero in ante lobortis"
        append_split = self.cleaner.prepare_sentence(sentence)
        assert append_split.shape[0] == SentenceCleaner.LENGTH

    def test_array_is_padded(self):
        sentence = "lorem ipsum dolor sit amet , consetetur"
        ar = self.cleaner.prepare_sentence(sentence)
        for i in range(8, 29):
            assert ar[i] == Vocabulary.PADDING


class DataLoaderTest(unittest.TestCase):
    if __name__ == '__main__':
        unittest.main()

    def setUp(self):
        self.loader = DataLoader("./test_sentences.txt", None, do_shuffle=False)

    def test_loadFileIntoMemory(self):
        # loading data is done in constructor
        assert self.loader.data is not None
        assert self.loader.data.shape == (6, 30)

    def test_iterate_over_all_epochs_and_batches(self):
        batches = self.loader.batch_iterator(3, 3)
        count = 0

        for i in batches:
            count += 1
            assert i.shape == (3, 30)
        assert count == 6


class VocabularyTest(unittest.TestCase):
    def setUp(self):
        self.voc = Vocabulary()
        Vocabulary.SIZE = 7

    def test_vocabulary(self):
        # data = [["eins", "zwei", "drei", "sieben"],["zwei", "zwei", "drei", "sechs"], ["drei", "drei", "eins", "eins"],["vier", "zwei", "eins"]]
        data = ["eins", "zwei", "drei", "sieben", "zwei", "zwei", "drei", "sechs", "drei", "drei", "eins", "eins",
                "vier", "zwei", "eins"]
        self.voc.extract(data)

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
