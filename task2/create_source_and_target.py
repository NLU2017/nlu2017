import os
import numpy as np

datadir = "./data/"
datafile = "Training_Shuffled_Dataset.txt"
output_source ="Training_Shuffled_Source.txt"
output_target = "Training_Shuffled_Target.txt"

read_path = os.path.join(datadir, datafile)
target_path = os.path.join(datadir, output_target)
source_path = os.path.join(datadir, output_source)

source = []
target = []
with open(read_path, "r") as file:
    for line in file:

        utterances = line.strip().split("\t")
        assert len(utterances) == 3
        source.append(utterances[0])
        source.append(utterances[1])
        target.append(utterances[1])
        target.append(utterances[2])

np.savetxt(source_path, np.asarray(source),
        fmt="%s",
        delimiter='\n')

np.savetxt(target_path, np.asarray(target),
        fmt="%s",
        delimiter='\n')
