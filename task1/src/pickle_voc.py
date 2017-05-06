from utils import Vocabulary
import pickle

vocabulary = Vocabulary()
vocabulary.load_file("../data/sentences.train")

vocabulary.get_vocabulary_as_dict()
vocabulary.get_inverse_voc_dict()

pickle.dump(vocabulary, open("../data/vocabulary.pickle", "wb"), pickle.HIGHEST_PROTOCOL)