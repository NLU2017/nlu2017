#!/bin/bash
# generate source and target files for the training data
DATA_PATH=./data
TRAIN_INPUT=$DATA_PATH/Training_Shuffled_Dataset.txt
VALIDATION_INPUT=$DATA_PATH/Validation_Shuffled_Dataset.txt
SEQ2SEQ_TOOLS=seq2seq/bin/tools/

echo " split triplets in training data"
python3 ./split_triplets.py --type=copy $TRAIN_INPUT

echo "split triplets in validation data"
python3 ./split_triplets.py --type=copy $VALIDATION_INPUT


#generate the vocabular from the source file (contains the same data in terms of words as the target file...)
MAX_VOCAB_SIZE=20000
VOCABULARY=$DATA_PATH/vocabulary_$MAX_VOCAB_SIZE.txt
MIN_FREQUENCY=0

echo "generating vocabulary with size $MAX_VOCAB_SIZE from training data"
#
sed "s/\t/' '/g" $DATA_PATH/Training_Shuffled_Dataset.txt | python3 $SEQ2SEQ_TOOLS/generate_vocab.py --max_vocab_size $MAX_VOCAB_SIZE --min_frequency $MIN_FREQUENCY  > $VOCABULARY



