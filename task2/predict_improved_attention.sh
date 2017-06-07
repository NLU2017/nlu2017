#!/bin/bash
export BASE_DIR=$(pwd)
export SEQ2SEQ_PATH=${BASE_DIR}/seq2seq
export DATA_DIR=${BASE_DIR}/data
export VOCAB_SOURCE=${DATA_DIR}/vocabulary_20000.txt
export VOCAB_TARGET=${DATA_DIR}/vocabulary_20000.txt
export TRAIN_SOURCES=${DATA_DIR}/Training_Shuffled_Dataset_source.txt
export TRAIN_TARGETS=${DATA_DIR}/Training_Shuffled_Dataset_target.txt
export DEV_SOURCES1=${DATA_DIR}/Validation_Shuffled_Dataset_source1.txt
export DEV_SOURCES2=${DATA_DIR}/Validation_Shuffled_Dataset_source2.txt
export DEV_TARGETS=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt

export DEV_TARGETS_REF=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt

export MODEL_DIR=${BASE_DIR}/runs/improved_attention

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python3 ${SEQ2SEQ_PATH}/bin/infer.py \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
   --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: DoubleSourceParallelTextInputPipeline
    params:
       source1_files:
        - $DEV_SOURCES1
       source2_files:
        - $DEV_SOURCES2
       target_files:
        - $DEV_TARGETS" \
  >  ${PRED_DIR}/predictions_baseline.txt
