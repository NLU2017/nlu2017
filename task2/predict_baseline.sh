#!/bin/bash
export BASE_DIR=$(pwd)
export SEQ2SEQ_PATH=${BASE_DIR}/seq2seq
export DATA_DIR=${BASE_DIR}/data
export VOCAB_SOURCE=${DATA_DIR}/vocabulary_20000.txt
export VOCAB_TARGET=${DATA_DIR}/vocabulary_20000.txt
export TRAIN_SOURCES=${DATA_DIR}/Training_Shuffled_Dataset_source.txt
export TRAIN_TARGETS=${DATA_DIR}/Training_Shuffled_Dataset_target.txt
export DEV_SOURCES=${DATA_DIR}/Validation_Shuffled_Dataset_source.txt
export DEV_TARGETS=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt
#export DEV_SOURCES=${DATA_DIR}/source_head.txt
#export DEV_TARGETS=${DATA_DIR}/target_head.txt

export DEV_TARGETS_REF=${DATA_DIR}/target_head.txt

export MODEL_DIR=${BASE_DIR}/runs/baseline

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python3 ${SEQ2SEQ_PATH}/bin/infer.py \
  --tasks "
    - class: DummyTask"\
  --model_dir $MODEL_DIR \
  --model_params "
    target.max_seq_len: 100
    vocab_source: $VOCAB_SOURCE
    vocab_target: $VOCAB_TARGET
    "\
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES
      target_files:
        - $DEV_TARGETS"\
  --batch_size 1 \
  > ${PRED_DIR}/utterance_perplexities.txt

python3 make_two_columns.py --input ${PRED_DIR}/utterance_perplexities.txt --output ${PRED_DIR}/perplexities.txt
