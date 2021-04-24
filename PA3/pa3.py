#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd











def main():
    ...



if __name__ == "__main__":
    if len(sys.argv) ==1:
        #main(['text.txt', 'B.txt', 'T.txt'])
        main(['text_V2.txt', 'B_V2.txt', 'T_V2.txt']) # B = context words, T = center words
    else:
        main(sys.argv[1:])
