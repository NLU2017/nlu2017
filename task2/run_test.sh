#!/bin/bash

# Declare variables
TEST_INPUT= $1
export PRED_DIR=pred_dir
export MODEL_DIR=runs/baseline_improved
export DATA_DIR=data
export TRAIN_SOURCES=${DATA_DIR}/Training_Shuffled_Dataset_source.txt

# Split Triplets of the test data
echo " split triplets in test data"
python3 ./split_triplets.py --type=copy --output_dir ${PRED_DIR} $TEST_INPUT

export TEST_SOURCE #some magic to get the names of the test_sources from $1
export TEST_TARGET # some magic for the targets.

# Run the model on the test data
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $TEST_SOURCE
       target_files:
        - $TEST_TARGET" \
  > ${PRED_DIR}/predictions.txt
