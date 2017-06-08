#!/bin/bash
export BASE_DIR=$(pwd)
export SEQ2SEQ_PATH=${BASE_DIR}/seq2seq
export DATA_DIR=${BASE_DIR}/data
export VOCAB_SOURCE=${DATA_DIR}/vocabulary_20000.txt
export VOCAB_TARGET=${DATA_DIR}/vocabulary_20000.txt
export TRAIN_SOURCES1=${DATA_DIR}/Training_Shuffled_Dataset_source1.txt
export TRAIN_SOURCES2=${DATA_DIR}/Training_Shuffled_Dataset_source2.txt
export TRAIN_TARGETS=${DATA_DIR}/Training_Shuffled_Dataset_target.txt
export DEV_SOURCES1=${DATA_DIR}/Validation_Shuffled_Dataset_source1.txt
export DEV_SOURCES2=${DATA_DIR}/Validation_Shuffled_Dataset_source2.txt
export DEV_TARGETS=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt

export DEV_TARGETS_REF=${DATA_DIR}/Validation_Shuffled_Dataset_target.txt
export TRAIN_STEPS=150000
export MODEL_DIR=${BASE_DIR}/runs/improved_attention

mkdir -p $MODEL_DIR


python3 ${SEQ2SEQ_PATH}/bin/train.py \
    --config_paths="
      ./train_improved_attention.yml,
      ${SEQ2SEQ_PATH}/example_configs/train_seq2seq.yml,
      ./text_metrics_perp.yml"\
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: DoubleSourceParallelTextInputPipeline
    params:
      source1_files:
        - $TRAIN_SOURCES1
      source2_files:
        - $TRAIN_SOURCES2
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: DoubleSourceParallelTextInputPipeline
    params:
       source1_files:
        - $DEV_SOURCES1
       source2_files:
        - $DEV_SOURCES2
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

echo "done"

