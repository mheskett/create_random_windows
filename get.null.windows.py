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
import seaborn as sns


### all files needed
fasta_fai = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa.fai"
blacklist = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.bed"
genome_fasta = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa"
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


def remove_blacklist(windows):
    blacklist = pybedtools.BedTool(blacklist)
    windows = windows.subtract(blacklist,A=True)

    return windows

def calculate_gc(windows):
    ### names=["#1_usercol", "2_usercol", "3_usercol",  "4_pct_at",  "5_pct_gc",  "6_num_A",    "7_num_C",  "8_num_G",  "9_num_T",   "10_num_N",  "11_num_oth",  "12_seq_len"]
    windows_nuc = windows.nucleotide_content(fi=genome_fasta)
    windows_nuc_df = windows_nuc.to_dataframe(names=["#1_usercol", "2_usercol", "3_usercol",  "4_pct_at",  "5_pct_gc",  "6_num_A",    "7_num_C",  "8_num_G",  "9_num_T",   "10_num_N",  "11_num_oth",  "12_seq_len"],

        disable_auto_names=True,header=None).drop(index=0)
    print(windows_nuc_df)
    plt.figure()
    sns.kdeplot([float(x) for x in windows_nuc_df["5_pct_gc"]],clip=(0,1))
    plt.xlim([0.2,0.91])
    plt.xticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    plt.savefig("test.pdf")
    return


calculate_gc(pybedtools.BedTool("/Users/heskett/breast.fragile.sites/reference_files/test.bed"))





