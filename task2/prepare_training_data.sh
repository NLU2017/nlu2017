#!/bin/bash
# generate source and target files for the training data
DATA_PATH=./data
TRAIN_INPUT=$DATA_PATH/Training_Shuffled_Dataset.txt
VALIDATION_INPUT=$DATA_PATH/Validation_Shuffled_Dataset.txt
SEQ2SEQ_TOOLS=seq2seq/bin/tools/

python ./split_triplets.py $TRAIN_INPUT

python ./split_triplets.py $VALIDATION_INPUT


#generate the vocabular from the source file (contains the same data in terms of words as the target file...)
#use MAX_VOCAB_SIZE=10000 as this is the vocabulary size used in the data (see Readme.txt)
MAX_VOCAB_SIZE=20000
MIN_FREQUENCY=0
#
python3 $SEQ2SEQ_TOOLS/generate_vocab.py --max_vocab_size $MAX_VOCAB_SIZE --min_frequency $MIN_FREQUENCY $DATA_PATH/Training_Shuffled_Dataset_source.txt


