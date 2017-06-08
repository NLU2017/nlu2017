import io
import argparse
import sys
import numpy as np


parser = argparse.ArgumentParser(
    description="Generate source and target files from the input files given (triplet per line separated by TAB. Both files will contain only 1 utterance per line: \n"
                "where the line k in the source is the input for the line k in target\n"
                "the original triplets are split into 3 files source1, source2 and target.\n"
                "each triplet expands over 2 lines:\n\n"
                "line 1: source1: <>  source2: s1 , target: s2\n"
                "line 2source1: s1   source2: s2, target: s3"
    )
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
    "--output_dir",
    dest="output_dir",
    default="./data",
    type=str,
    help="path to the output directory (default: ./data)")

parser.add_argument(
    "--fix_prefix",
    dest="fix_prefix",
    default="",
    type=str,
    help="If given, this fix prefix is being used")

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

if args.fix_prefix is not "":
    prefix = args.fix_prefix
source1_filename = args.output_dir + "/" + prefix+ "_source_d1.txt"
source2_filename = args.output_dir + "/" + prefix+ "_source_d2.txt"
target_filename = args.output_dir + "/" + prefix+ "_target_d.txt"

source1 = []
source2 = []
target = []
max_length = 0
for line in args.infile:
    utterances = line.strip().split("\t")
    assert len(utterances) == 3
    #first line
    source1.append("")
    source2.append(utterances[0])
    target.append(utterances[1])
    #second line
    source1.append(utterances[0])
    source2.append(utterances[1])
    target.append(utterances[2])

print("max length of source {}".format(find_max_length_in_words(source2)))
print ("max length of targets: {}".format(find_max_length_in_words(target)))

with io.open(source1_filename, "w", encoding='utf8') as source_file:
    for record in source1:
        source_file.write(record + "\n")

print("done writing {}".format(source1_filename))


with io.open(source2_filename, "w", encoding='utf8') as source_file:
    for record in source2:
        source_file.write(record + "\n")

print("done writing {}".format(source2_filename))


with io.open(target_filename, "w", encoding='utf8') as target_file:
    for record in target:
        target_file.write(record + "\n")


print("done writing {}".format(target_filename))