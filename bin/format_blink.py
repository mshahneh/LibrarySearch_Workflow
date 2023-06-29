#!/usr/bin/python


import sys
import getopt
import os
import pandas as pd
from collections import defaultdict
import argparse
import glob

def _parse_file(input_filename):
    # checking extension
    if not input_filename.endswith(".tsv"):
        df = pd.read_csv(input_filename, sep="\t")
    elif input_filename.endswith(".csv"):
        df = pd.read_csv(input_filename, sep=",")

    return df


def main():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Formatting data results')
    parser.add_argument('input_blink_file', help='input_blink_file')
    parser.add_argument('output_file', help='output_file')

    args = parser.parse_args()

    df = _parse_file(args.input_blink_file)

    print(df.columns)

if __name__ == "__main__":
    main()
