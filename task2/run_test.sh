#!/bin/bash

# Declare variables
TEST_INPUT=$1
export PRED_DIR=pred_dir
export MODEL_DIR=runs/improved_attention
export DATA_DIR=data
export TRAIN_SOURCES=${DATA_DIR}/Training_Shuffled_Dataset_source.txt
export VOCAB_SOURCE=${DATA_DIR}/vocabulary_20000.txt
export VOCAB_TARGET=${DATA_DIR}/vocabulary_20000.txt

echo "creating prediction directory"
mkdir -p $PRED_DIR

# Split Triplets of the test data
echo " split triplets in test data from $TEST_INPUT"
python3 ./split_triplets_for_double_source.py --output_dir ${PRED_DIR} --fix_prefix test_data $TEST_INPUT
#python3 ./split_triplets_for_double_source.py --output_dir pred_dir --fix_prefix test_data data/Validation_Shuffled_Dataset.txt


export TEST_SOURCE1=${PRED_DIR}/test_data_source_d1.txt
export TEST_SOURCE2=${PRED_DIR}/test_data_source_d2.txt
export TEST_TARGET=${PRED_DIR}/test_data_target_d.txt

echo "start evalution "
# Run the model on the test data
python -m bin.infer \
  --tasks "
    - class: GetPerplexity" \
  --model_dir $MODEL_DIR \
  --model_params "
    target.max_seq_len: 100
    vocab_source: $VOCAB_SOURCE
    vocab_target: $VOCAB_TARGET
    "\
  --input_pipeline "
    class: DoubleSourceParallelTextInputPipeline
    params:
      source1_files:
        - $TEST_SOURCE1
      source2_files:
        - $TEST_SOURCE2
      target_files:
        - $TEST_TARGET" \
  > ${PRED_DIR}/utterance_perplexities.txt

  python3 make_two_columns.py --input ${PRED_DIR}/utterance_perplexities.txt --output ${PRED_DIR}/perplexities.txt
  echo "DONE: wrote perplexities to ${PRED_DIR}/perplexities.txt"
