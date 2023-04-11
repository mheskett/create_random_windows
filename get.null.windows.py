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
blacklist_file = "/Users/heskett/breast.fragile.sites/reference_files/hg19-blacklist.v2.bed"
genome_fasta = "/Users/heskett/breast.fragile.sites/reference_files/genome.fa"
common_snps = "/Users/heskett/breast.fragile.sites/reference_files/ucsc.common.snps.nochr.bed"

###


def make_windows(length, overlap_fraction=0.25, write_file=False, file_path=None):
    a=pybedtools.BedTool()
    windows=a.window_maker(g=fasta_fai, w=length, s=length*window_fraction)

    return windows


def sample_windows(number, window_object, fraction_windows=None, window_file=None):
    if number > len(window_object):
        print("ERROR: trying to sample more windows than available")
        return

    return 


def random_windows(length,number, write_file=False, file_path=None):
    a=pybedtools.BedTool()
    windows = a.random(l=length, n=number, g=fasta_fai)

    return windows


def remove_blacklist(windows):
    blacklist = pybedtools.BedTool(blacklist_file)
    windows = windows.subtract(blacklist, A=True)

    return windows

def calculate_gc(windows):
    # can either chop at chrom start stop, or keep everything and then create the header basd on how long
    # names=["#1_usercol", "2_usercol", "3_usercol",  "4_pct_at",  "5_pct_gc",  "6_num_A",    "7_num_C",  "8_num_G",  "9_num_T",   "10_num_N",  "11_num_oth",  "12_seq_len"],
    num_cols = len(windows.to_dataframe().columns)
    windows_nuc = windows.nucleotide_content(fi=genome_fasta)
    windows_nuc_df = windows_nuc.to_dataframe(

        disable_auto_names=True)
    print(windows_nuc_df)
    plt.figure()
    sns.kdeplot([float(x) for x in windows_nuc_df[str(num_cols+2)+"_pct_gc"]], clip=(0, 1))
    plt.xlim([0,1])
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.savefig("gc.pdf")

    return

def common_snp_density(windows):
    num_cols = len(windows.to_dataframe().columns)

    snps = pybedtools.BedTool(common_snps)
    window_snps_df = windows.intersect(snps,c=True).to_dataframe(disable_auto_names=True,header=None)
    print(window_snps_df)
    window_snps_df["snp_density"] = window_snps_df[num_cols].astype(int) / ((window_snps_df[2].astype(int) - window_snps_df[1].astype(int)) / 1000)
    
    plt.figure()
    sns.kdeplot(window_snps_df["snp_density"], clip=(0, 10),label="snps_per_kb")
    plt.xlim([0,10])
    plt.xticks(list(range(0,11)))
    plt.suptitle("snps_per_kb")
    plt.savefig("snps_per_kb.pdf")

    return


#calculate_gc(remove_blacklist(random_windows(100000,1000))) # dont forget random seeds

#calculate_gc(pybedtools.BedTool("/Users/heskett/breast.fragile.sites/reference_files/test.bed"))


common_snp_density(random_windows(20000,500))


