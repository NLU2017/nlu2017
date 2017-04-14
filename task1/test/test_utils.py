import sys
sys.path.append('../src')
import unittest
from utils import SentenceCleaner

class SentenceCleanerTest(unittest.TestCase):

     if __name__ == '__main__':
          unittest.main()

     def test_SentenceCleaner_returns_array_with_shape_30(self):
          cleaner = SentenceCleaner()
          sentence = "lorem ipsum dolor sit amet , consetetur"
          append_split = cleaner.prepare_sentence(sentence)

          assert append_split.shape[0] == cleaner.LENGTH
          assert append_split[-1] == cleaner.END_SEQ


     def test_SentenceCleaner_array_starts_with_bos(self):
          sentence = "lorem ipsum dolor sit amet , consetetur"
          cleaner = SentenceCleaner()
          append_split = cleaner.prepare_sentence(sentence)

          assert append_split[0] == cleaner.INIT_SEQ
          assert append_split[-1] == cleaner.END_SEQ


     def test_SentenceCleaner_array_ends_with_is_eos(self):
          sentence = "lorem ipsum dolor sit amet , consetetur"
          cleaner = SentenceCleaner()
          append_split = cleaner.prepare_sentence(sentence)
          assert append_split[-1] == cleaner.END_SEQ


