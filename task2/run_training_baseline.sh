#!/bin/bash
export BASE_DIR=$(pwd)
export SEQ2SEQ_PATH=${BASE_DIR}/seq2seq/
export DATA_DIR=${BASE_DIR}/data
export VOCAB_SOURCE=${DATA_DIR}/vocabulary_20000.txt
export VOCAB_TARGET=${DATA_DIR}/vocabulary_20000.txt
export TRAIN_SOURCES=${DATA_DIR}/Training_Shuffled_Dataset_source.txt
export TRAIN_TARGETS=${DATA_DIR}/Training_Shuffled_Dataset_target.txt
export DEV_SOURCES=${DATA_DIR}/Validation_Shuffled_Dataset_source.txt
export DEV_TARGETS=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt

export DEV_TARGETS_REF=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt
export TRAIN_STEPS=1000
export MODEL_DIR=${BASE_DIR}/runs/baseline

mkdir -p $MODEL_DIR


python3 -m bin.train \
    --config_paths="
      ./train_baseline.yml,
      ${SEQ2SEQ_PATH}/example_configs/train_seq2seq.yml"\
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

echo "done"
