#!/bin/bash
export BASE_DIR=$(pwd)
export SEQ2SEQ_PATH=${BASE_DIR}/seq2seq
export VOCAB=vocabulary_20000.txt

export MODEL_DIR=${BASE_DIR}/improved_attention

export PRED_DIR=pred
mkdir -p ${PRED_DIR}

echo " split triplets in training data"
python3 ./split_triplets_for_double_source.py --output_dir=./ --fix_prefix test_data $1

python3 ${SEQ2SEQ_PATH}/bin/infer.py \
  --tasks "
    - class: DummyTask" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: DoubleSourceParallelTextInputPipeline
    params:
       source1_files:
        - test_data_source_d1.txt
       source2_files:
        - test_data_source_d2.txt
       target_files:
        - test_data_target_d.txt" \
  >  ${PRED_DIR}/predictions_improved_attention.txt

python3 make_two_columns.py --input ${PRED_DIR}/predictions_improved_attention.txt --output ${PRED_DIR}/perplexities.txt