import sys
sys.path.append('../src')
from utils import SentenceCleaner

def test_prepare_sentence():
     sentence = "lorem ipsum dolor sit amet , consetetur"

     cleaner = SentenceCleaner()

     append_split = cleaner.prepare(sentence)

     assert len(append_split) == cleaner.LENGTH
     assert append_split[0] == cleaner.INIT_SEQ
     assert append_split[-1] == cleaner.END_SEQ



