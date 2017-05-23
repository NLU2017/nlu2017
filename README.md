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
   
### get started:
Switch to the master branch in the project directory and do a 
```
> git checkout master
> git pull
```
you should now have a folder `task2` that contains
  - `data`: the data directory, copy the data for task2 to this directory and extract it here. The data files should not be added
  to the Version control, there is a .gitignore which should take care of this...
  - the scripts `prepare_training_data.sh`, `run_training_baseline.sh`, `split_triplets.py`, `train_baseline.yml`

change into the task2 directory and clone the seq2seq model from [github](https://google.github.io/seq2seq/)
then do the [following](https://google.github.io/seq2seq/getting_started/)

```bash
> git clone https://github.com/google/seq2seq.git
> cd seq2seq

# Install package and dependencies
pip install -e .

```
maybe you have to sudo for the pip install, I had to on the Azure VM.
 The model uses `yml` configuration files, I had 
to install the corresponding python module:
```bash
> pip install pyyaml

```
you should then be able to run the unittests for the seq2seq model.

```
python -m unittest seq2seq.test.pipeline_test
```


For training the baselin model do the following:
```bash
> ./prepare_training_data.sh
```
this scripts creates a source file an a target file out of the triplets. It also outputs the maximal sequence length in the
training data and generates a vocabulary by using the script from `seq2seq/bin/tools/generate_vocab.py`
you can tune the number of words in the vocabulary and whether the target sequences should be reversed. See the script for details.

With this run 
```
> run_training_baseline.sh
```
to start the training. Details for the baseline configuration are in `train_baseline.yml` it starts a Vanilla seq2seq model
with the input data that we have generated before. I use reversed target data, and the BasicLSTMCell. I took the Encoder and Decoder 
values from [source code of the basic model](https://github.com/google/seq2seq/blob/master/seq2seq/models/basic_seq2seq.py) 
Don't know whether this makes sense. 

There are some more model in the seq2seq Repo see the [Doku](https://google.github.io/seq2seq/models/#seq2seqmodel) and 
the [translation tutorial](https://google.github.io/seq2seq/nmt/) for the config values.

###next steps
There is already an Attention model `AttentionSeq2Seq`.
For our own model we will have to write a model class which inherits from `ModelBase` and `Seq2SeqModel`. That is how I understood the thing.
See [models reference](https://google.github.io/seq2seq/models/).
Happy coding...


## Contacts
mark.bosshard@uzh.ch

oliver.burkhard@gmail.com

sijha@student.ethz.ch

luzm@student.ethz.ch
