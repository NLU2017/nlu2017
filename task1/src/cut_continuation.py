import os
from utils import Vocabulary
import numpy.random as rd
max_length = 18
input_path = "../data/sentences.continuation"
output_path = "../data/sentences.mycontinuation"

outputfile = open(output_path, "w")

with open(input_path) as file:

    for line in file:
        words = line.strip().split(Vocabulary.SPLIT)
        le = rd.randint(1, max_length)
        out = Vocabulary.SPLIT.join(words[0:le])
        print(out)
        outputfile.write(out+"\n")

outputfile.close()

