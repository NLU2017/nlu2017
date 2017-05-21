# nlu2017

Project repository for the lecture Natural Language Understanding at ETH 2017.

## Contents

### Structure
 - `task1`:  contains all code relevant for the first task.
    - `data`: data files, these files are not checked into the repository, download to your machine directly from polybox..
    - `src`: all (productive) python src codes
    - `test`: unit tests

##Technical requirements
the code uses the following libraries:
 - python 3.5
 - tensorflow
 - numpy
 - unittest: for running the unit tests in the test directory


##Task 2: Baseline
I took the vanilla seq2seq model from [google.github.io/seq2seq](https://google.github.io/seq2seq/). That is the `BasicSeq2Seq` model with 
the parameters as suggested in the source code (see `train_baseline.yml`)


### structure
 -`task2` : contains all our scripts and configuration files.
   - `prepare_training_data.sh`: calls the `split_triplet.py`script to generate separate files for source and target than can be used 
   by the model. There is an option to reverse the targets.
   
   - `data`: contains the data we were given by the tutors also the generated source, target and vocabulary files are stored there.
   _please do not checkin anything into this folder_ all files can be regenerated from the original data
   - `seq2seq`: root of the repo cloned from [google.github.io/seq2seq](https://google.github.io/seq2seq/). 
   - `runs`: is created by the `run_training_baseline.sh` and contains subdirectories for training runs, checkpoints, summaries, etc
   



## Contacts
mark.bosshard@uzh.ch

oliver.burkhard@gmail.com

sijha@student.ethz.ch

luzm@student.ethz.ch
