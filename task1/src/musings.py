#vm login
# nlu17eth
# nlu17group25@eth

#azure portal
#nlu17eth@outlook.com
#nlu17group25

from utils import DataLoader
import pickle
vocabulary = pickle.load(open("./vocabulary.pickle", "rb"))
eval_loader = DataLoader("../data/sentences_test",
                             vocabulary, do_shuffle=False)
batches_eval = eval_loader.batch_iterator(num_epochs=1, batch_size=10)
for data_eval in batches_eval:
    print(data_eval)