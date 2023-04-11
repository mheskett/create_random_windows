import os
import csv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pybedtools
import re
import time
import scipy.stats
import multiprocessing as mp
import random
import json


### all files needed
fasta_fai = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa.fai"
blacklist = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.bed"
###


def make_windows(length, overlap_fraction=0.25, write_file=False, file_path=None):
    a=pybedtools.BedTool()
    windows=a.window_maker(g=fasta_fai, w=length, s=length*window_fraction)

    return windows


def sample_windows(number,window_object,fraction_windows=None,window_file=None):
    if number > len(window_object):
        print("ERROR: trying to sample more windows than available")
        return

    return 


def random_windows(length,number, write_file=False, file_path=None):
    a=pybedtools.BedTool()
    windows = a.random(l=length, n=number, genome=fasta_fai)

    return windows


def filter_windows(windows):
    blacklist = pybedtools.BedTool(blacklist)
    windows = windows.subtract(blacklist,A=True)

    return windows

