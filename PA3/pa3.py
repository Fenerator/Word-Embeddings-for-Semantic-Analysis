#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=int, help="pa3_T.txt file", required=True)
    parser.add_argument("--B", action="store_true", help="pa3_B.txt file", required=True)
    parser.add_argument("--text", type=str, help="Input txt file", required=True)

    args = parser.parse_args()

    return args

def get_sparse():
    """returns PMI weighted coocurrence matrix from PA1"""


def get_dense():
    """returns word2vec representation"""
    ...

def single_evaluation():
    ...

def cross_validation_eval():
    ...












def main():
    args = parse_args()

    x = args.tokenize









if __name__ == "__main__":
    main()

