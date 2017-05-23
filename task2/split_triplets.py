import io
import argparse
import sys
import numpy as np


parser = argparse.ArgumentParser(
    description="Generate source and target files from the input files given (triplet per line separated by TAB. Both files will contain only 1 utterance per line: \n"
                "where the line k in the source is the input for the line k in target\n"
                "the original triplets are split: 1 (source)-> 2 (target), 2(source) -> 3(target)")
parser.add_argument(
    "infile",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="Input tokenized text file to be processed.")
parser.add_argument(
    "--delimiter",
    dest="delimiter",
    type=str,
    default="\t",
    help="Delimiter that separates the triplets in the orginal file. (default \"\\t\")"
)
parser.add_argument(
    "--vocab_size",
    type=int,
    default=10000,
    help="size of the vocabulary (default: 10000)")
parser.add_argument(
    "--type",
    type=str,
    default="copy",
    choices=["copy", "reverse"],
    help="Type of dataet to generate. One of \"copy\" (take the sentences as they are in the input file) or \"reverse\" (reverse the sentences in the target file) (default: copy) ")
parser.add_argument(
    "--output_dir",
    dest="output_dir",
    default="./data",
    type=str,
    help="path to the output directory (default: ./data)")
args = parser.parse_args()


def find_max_length_in_words(sentences):
    length = 0
    for s in sentences:
        le = len(s.split(" "))
        if le > length:
            length = le
    return length


fname = args.infile.name.split("/")[-1]
prefix = fname.split(".")
if len(prefix) > 1:
    prefix = prefix[-2]
source_filename = args.output_dir + "/" + prefix+ "_source.txt"
target_filename = args.output_dir + "/" + prefix+ "_target.txt"

source = []
target = []
max_length = 0
for line in args.infile:
    utterances = line.strip().split("\t")
    assert len(utterances) == 3
    source.append(utterances[0])
    source.append(utterances[1])
    target.append(utterances[1])
    target.append(utterances[2])

print("max length of source {}".format(find_max_length_in_words(source)))
print ("max length of targets: {}".format(find_max_length_in_words(target)))


if args.type == 'reverse':
    for (i, record) in enumerate(source):
        source[i] = " ".join(reversed(record.split(" ")))

with io.open(source_filename, "w", encoding='utf8') as source_file:
    for record in source:
        source_file.write(record + "\n")

print("done writing {}".format(source_filename))



with io.open(target_filename, "w", encoding='utf8') as target_file:
    for record in target:
        target_file.write(record + "\n")



print("done writing {}".format(source_filename))
