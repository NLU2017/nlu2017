import io
import argparse
import numpy as np
from math import floor

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="runs/baseline/pred/utterance_perplexities.txt",
    help="Input file containing utterance-level perplexities")

parser.add_argument(
    "--output",
    type=str,
    default="runs/baseline/pred/perplexities.txt",
    help="Name of the output file")

args = parser.parse_args()

inp = args.input
outp = args.output

content = open(inp).readlines()
content_numeric = [float(x.rstrip()) for x in content if is_number(x.rstrip())]
assert(len(content_numeric) % 2 == 0)
output = np.empty([floor(len(content_numeric) / 2), 2])
for i in range(len(content_numeric)):
    output[i // 2, i % 2] = content_numeric[i]

np.savetxt(outp, output)
